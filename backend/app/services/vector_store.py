from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import chromadb

from backend.app.services.chunker import chunk_abstract
from backend.app.services.embedding import EmbeddingClient
from src.classification.paper_classifier import ClassifiedPaper

logger = logging.getLogger(__name__)

# Dimension of text-embedding-3-small; used as placeholder for non-embeddable chunks.
_EMBEDDING_DIM = 1536
_PLACEHOLDER_EMBEDDING = [0.0] * _EMBEDDING_DIM


@dataclass
class StoreResult:
    chunks_stored: int
    chunks_skipped: int
    papers_processed: int


@dataclass
class ChunkResult:
    chunk_text: str
    chunk_index: int
    distance: float
    paper_id: str
    title: str
    year: str
    citation_count: int
    source: str
    classification: str
    embeddable: bool


def embed_and_store(
    papers: list[ClassifiedPaper],
    session_id: UUID | str,
    chroma: chromadb.HttpClient,
    embedder: EmbeddingClient | None = None,
) -> StoreResult:
    if embedder is None:
        embedder = EmbeddingClient()

    coll_name = str(session_id)
    collection = chroma.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        logger.info(
            "Collection %s already populated (%d chunks), reusing.", coll_name, collection.count()
        )
        return StoreResult(chunks_stored=0, chunks_skipped=0, papers_processed=len(papers))

    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []
    embeddable_positions: list[int] = []

    for paper in papers:
        p = paper.paper
        year = p.published[:4] if p.published else ""
        for chunk in chunk_abstract(p.abstract):
            pos = len(all_ids)
            all_ids.append(f"{p.id}__chunk_{chunk.chunk_index}")
            all_documents.append(chunk.text)
            all_metadatas.append(
                {
                    "paper_id": p.id,
                    "title": p.title,
                    "year": year,
                    "citation_count": p.citation_count or 0,
                    "source": p.source,
                    "classification": paper.category.value,
                    "chunk_index": chunk.chunk_index,
                    "embeddable": chunk.embeddable,
                }
            )
            if chunk.embeddable:
                embeddable_positions.append(pos)

    all_embeddings: list[list[float]] = [_PLACEHOLDER_EMBEDDING] * len(all_ids)

    if embeddable_positions:
        texts = [all_documents[i] for i in embeddable_positions]
        raw = embedder.embed_texts(texts)
        for local_i, global_i in enumerate(embeddable_positions):
            if raw[local_i] is not None:
                all_embeddings[global_i] = raw[local_i]
            # On None: leave placeholder in place; the storage loop detects it below.

    chunks_stored = 0
    chunks_skipped = 0
    store_ids, store_docs, store_metas, store_embeds = [], [], [], []

    for doc_id, doc, meta, emb in zip(all_ids, all_documents, all_metadatas, all_embeddings):
        if not meta["embeddable"]:
            # Short/empty abstract: store with placeholder so metadata is preserved;
            # excluded from similarity search via where={"embeddable": True}.
            store_ids.append(doc_id)
            store_docs.append(doc)
            store_metas.append(meta)
            store_embeds.append(emb)
            chunks_stored += 1
        elif emb is _PLACEHOLDER_EMBEDDING:
            # Embeddable chunk whose embedding API call failed: skip, do not store.
            logger.warning("Skipping chunk %s: embedding failed.", doc_id)
            chunks_skipped += 1
        else:
            store_ids.append(doc_id)
            store_docs.append(doc)
            store_metas.append(meta)
            store_embeds.append(emb)
            chunks_stored += 1

    if store_ids:
        collection.upsert(
            ids=store_ids,
            documents=store_docs,
            metadatas=store_metas,
            embeddings=store_embeds,
        )

    return StoreResult(
        chunks_stored=chunks_stored,
        chunks_skipped=chunks_skipped,
        papers_processed=len(papers),
    )


def search_similar(
    session_id: str,
    query: str,
    top_k: int,
    chroma: chromadb.HttpClient,
    embedder: EmbeddingClient,
) -> list[ChunkResult]:
    try:
        collection = chroma.get_collection(name=session_id)
    except Exception:
        return []

    total = collection.count()
    if total == 0:
        return []

    query_embedding = embedder.embed_query(query)

    n = min(top_k, total)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where={"embeddable": True},
        include=["documents", "metadatas", "distances"],
    )

    results: list[ChunkResult] = []
    if not raw["ids"] or not raw["ids"][0]:
        return results

    for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        results.append(
            ChunkResult(
                chunk_text=doc,
                chunk_index=meta["chunk_index"],
                distance=dist,
                paper_id=meta["paper_id"],
                title=meta["title"],
                year=meta["year"],
                citation_count=meta["citation_count"],
                source=meta["source"],
                classification=meta["classification"],
                embeddable=meta["embeddable"],
            )
        )

    return results
