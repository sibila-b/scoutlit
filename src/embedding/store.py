from __future__ import annotations

import logging

import chromadb

from src.classification.paper_classifier import ClassifiedPaper
from src.embedding.chunker import chunk_text
from src.embedding.embedder import EmbeddingClient

logger = logging.getLogger(__name__)

_MIN_ABSTRACT_CHARS = 20
_PLACEHOLDER_DIM = 1024


def _make_collection_name(session_id: str) -> str:
    return f"session-{session_id}"


class VectorStore:
    def __init__(
        self,
        chroma_client: chromadb.HttpClient,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._chroma = chroma_client
        self._embedder = embedding_client or EmbeddingClient()

    def get_or_create_collection(self, session_id: str) -> chromadb.Collection:
        return self._chroma.get_or_create_collection(
            name=_make_collection_name(session_id),
            metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self, session_id: str) -> bool:
        try:
            col = self._chroma.get_collection(_make_collection_name(session_id))
            return col.count() > 0
        except Exception:
            return False

    def embed_and_store(
        self,
        session_id: str,
        papers: list[ClassifiedPaper],
        skip_if_exists: bool = True,
    ) -> dict[str, int]:
        if skip_if_exists and self.collection_exists(session_id):
            logger.info("Collection for session %s already exists — reusing.", session_id)
            return {"stored": 0, "skipped_short": 0, "skipped_error": 0, "reused": 1}

        collection = self.get_or_create_collection(session_id)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        stored = 0
        skipped_short = 0
        skipped_error = 0

        for classified in papers:
            paper = classified.paper
            abstract = paper.abstract.strip()

            if len(abstract) < _MIN_ABSTRACT_CHARS:
                logger.warning(
                    "Abstract for paper %s is under %d chars — storing metadata only.",
                    paper.id,
                    _MIN_ABSTRACT_CHARS,
                )
                collection.upsert(
                    ids=[f"{paper.id}::meta"],
                    embeddings=[[0.0] * _PLACEHOLDER_DIM],
                    documents=[""],
                    metadatas=[
                        {
                            "paper_id": paper.id,
                            "title": paper.title,
                            "year": paper.year,
                            "source": paper.source,
                            "classification": classified.category.value,
                            "chunk_index": 0,
                            "embeddable": False,
                        }
                    ],
                )
                skipped_short += 1
                continue

            chunks = chunk_text(abstract)
            chunk_embeddings = self._embedder.embed_batch(chunks)

            for chunk_index, (chunk, embedding) in enumerate(
                zip(chunks, chunk_embeddings, strict=True)
            ):
                if embedding is None:
                    logger.warning(
                        "Embedding failed for paper %s chunk %d — skipping.",
                        paper.id,
                        chunk_index,
                    )
                    skipped_error += 1
                    continue

                ids.append(f"{paper.id}::{chunk_index}")
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append(
                    {
                        "paper_id": paper.id,
                        "title": paper.title,
                        "year": paper.year,
                        "source": paper.source,
                        "classification": classified.category.value,
                        "chunk_index": chunk_index,
                        "embeddable": True,
                    }
                )
                stored += 1

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Session %s: stored %d chunks, %d short, %d errors.",
                session_id,
                stored,
                skipped_short,
                skipped_error,
            )

        return {"stored": stored, "skipped_short": skipped_short, "skipped_error": skipped_error}

    def similarity_search(
        self,
        session_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        if not self.collection_exists(session_id):
            return []

        query_embedding = self._embedder.embed_single(query)
        if query_embedding is None:
            raise RuntimeError("Failed to embed query — cannot perform similarity search.")

        collection = self._chroma.get_collection(_make_collection_name(session_id))
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            chunks.append(
                {
                    "text": doc,
                    "score": round(1 - dist, 4),
                    "paper_id": meta["paper_id"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "source": meta["source"],
                    "classification": meta["classification"],
                    "chunk_index": meta["chunk_index"],
                    "embeddable": meta.get("embeddable", True),
                }
            )

        return chunks
