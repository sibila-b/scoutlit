from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import pytest

from src.classification.paper_classifier import ClassifiedPaper, PaperCategory
from src.retrieval.arxiv_client import Paper


# ── in-memory ChromaDB stub ────────────────────────────────────────────────────

class _FakeCollection:
    """In-memory collection with real cosine-distance search."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: list[list[float]] = []

    def count(self) -> int:
        return len(self._ids)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            if doc_id in self._ids:
                pos = self._ids.index(doc_id)
                self._documents[pos] = doc
                self._metadatas[pos] = meta
                self._embeddings[pos] = emb
            else:
                self._ids.append(doc_id)
                self._documents.append(doc)
                self._metadatas.append(meta)
                self._embeddings.append(emb)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        q = query_embeddings[0]

        def _cosine_dist(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 1.0
            return 1.0 - dot / (mag_a * mag_b)

        def _matches(meta: dict, clause: dict | None) -> bool:
            if clause is None:
                return True
            return all(meta.get(k) == v for k, v in clause.items())

        scored = [
            (i, _cosine_dist(q, self._embeddings[i]))
            for i in range(len(self._ids))
            if _matches(self._metadatas[i], where)
        ]
        scored.sort(key=lambda x: x[1])
        top = scored[:n_results]

        return {
            "ids": [[self._ids[i] for i, _ in top]],
            "documents": [[self._documents[i] for i, _ in top]],
            "metadatas": [[self._metadatas[i] for i, _ in top]],
            "distances": [[d for _, d in top]],
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self._cols: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict | None = None) -> _FakeCollection:
        if name not in self._cols:
            self._cols[name] = _FakeCollection()
        return self._cols[name]

    def get_collection(self, name: str) -> _FakeCollection:
        if name not in self._cols:
            raise ValueError(f"Collection '{name}' not found")
        return self._cols[name]


# ── test helpers ───────────────────────────────────────────────────────────────

DIM = 1536


def _unit_vec(hot_dim: int) -> list[float]:
    v = [0.0] * DIM
    v[hot_dim] = 1.0
    return v


def _paper(pid: str, title: str, abstract: str) -> ClassifiedPaper:
    return ClassifiedPaper(
        paper=Paper(
            id=pid,
            title=title,
            authors=["Author A"],
            abstract=abstract,
            published="2024-01-01T00:00:00",
            url=f"https://arxiv.org/abs/{pid}",
            citation_count=10,
        ),
        category=PaperCategory.RECENT,
        rationale="Recent work.",
    )


@pytest.fixture()
def three_papers() -> list[ClassifiedPaper]:
    return [
        _paper(
            "p1", "Transformers for NLP",
            "We introduce a self-attention mechanism for sequence modelling in natural language "
            "processing tasks, enabling the model to capture long-range dependencies efficiently. " * 3,
        ),
        _paper(
            "p2", "Graph Neural Networks",
            "Graph neural networks propagate information along edges in a graph structure to learn "
            "rich node embeddings for downstream classification and regression tasks. " * 3,
        ),
        _paper(
            "p3", "Diffusion Models",
            "Denoising diffusion probabilistic models learn to reverse a gradual noising process "
            "applied to data, generating high-quality samples in a wide variety of domains. " * 3,
        ),
    ]


# ── integration tests ──────────────────────────────────────────────────────────

def test_embed_and_store_then_search_returns_expected_top_result(
    three_papers: list[ClassifiedPaper],
) -> None:
    """Full pipeline: embed_and_store 3 papers → search_similar returns the expected top paper."""
    from backend.app.services.vector_store import embed_and_store, search_similar

    session_id = str(uuid.uuid4())

    # Each paper gets an orthogonal unit-vector embedding.
    # The query is aligned with p1 (dim 0), so p1 must rank first.
    emb_map = {
        "self-attention": _unit_vec(0),   # p1
        "Graph neural": _unit_vec(1),     # p2
        "Denoising diffusion": _unit_vec(2),  # p3
    }

    def fake_embed_texts(texts: list[str]) -> list[list[float] | None]:
        result = []
        for t in texts:
            matched = next((v for k, v in emb_map.items() if k in t), _unit_vec(10))
            result.append(matched)
        return result

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.side_effect = fake_embed_texts
    mock_embedder.embed_query.return_value = _unit_vec(0)  # closest to p1

    chroma = _FakeChromaClient()

    store = embed_and_store(
        papers=three_papers,
        session_id=session_id,
        chroma=chroma,
        embedder=mock_embedder,
    )

    assert store.papers_processed == 3
    assert store.chunks_stored > 0
    assert store.chunks_skipped == 0

    results = search_similar(
        session_id=session_id,
        query="self-attention NLP",
        top_k=3,
        chroma=chroma,
        embedder=mock_embedder,
    )

    assert len(results) == 3
    assert results[0].paper_id == "p1", (
        f"Expected p1 first; got {[(r.paper_id, r.distance) for r in results]}"
    )

    top = results[0]
    assert top.title == "Transformers for NLP"
    assert top.year == "2024"
    assert top.citation_count == 10
    assert top.source == "arxiv"
    assert top.classification == "recent"
    assert top.embeddable is True


def test_reuses_existing_collection_without_re_embedding(
    three_papers: list[ClassifiedPaper],
) -> None:
    from backend.app.services.vector_store import embed_and_store

    session_id = str(uuid.uuid4())
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [_unit_vec(0)] * 100

    chroma = _FakeChromaClient()
    embed_and_store(three_papers, session_id, chroma, mock_embedder)
    first_call_count = mock_embedder.embed_texts.call_count

    embed_and_store(three_papers, session_id, chroma, mock_embedder)
    assert mock_embedder.embed_texts.call_count == first_call_count


def test_short_abstract_not_included_in_similarity_results() -> None:
    from backend.app.services.vector_store import embed_and_store, search_similar

    session_id = str(uuid.uuid4())
    short_paper = _paper("px", "Tiny", "Too short.")  # < 20 chars, embeddable=False

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = []
    mock_embedder.embed_query.return_value = _unit_vec(0)

    chroma = _FakeChromaClient()
    result = embed_and_store([short_paper], session_id, chroma, mock_embedder)

    assert result.papers_processed == 1
    # Non-embeddable chunk is skipped from the store (embeddable=False is filtered at query time)
    results = search_similar(session_id, "anything", 5, chroma, mock_embedder)
    assert results == []


def test_failed_embedding_chunk_is_skipped() -> None:
    from backend.app.services.vector_store import embed_and_store

    session_id = str(uuid.uuid4())
    paper = _paper(
        "p9", "Some Paper",
        "This abstract is long enough to be embeddable and contains useful content. " * 3,
    )

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [None]  # simulate embedding failure

    chroma = _FakeChromaClient()
    result = embed_and_store([paper], session_id, chroma, mock_embedder)

    assert result.chunks_skipped >= 1
    assert result.chunks_stored == 0
