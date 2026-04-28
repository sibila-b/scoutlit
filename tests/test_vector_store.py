from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.classification.paper_classifier import ClassifiedPaper, PaperCategory
from src.embedding.store import VectorStore
from src.models.paper import PaperResult


def _make_paper(paper_id: str, abstract: str, source: str = "arxiv") -> ClassifiedPaper:
    return ClassifiedPaper(
        paper=PaperResult(
            id=paper_id,
            title=f"Paper {paper_id}",
            authors=["Author A"],
            abstract=abstract,
            year="2024",
            url=f"https://arxiv.org/abs/{paper_id}",
            source=source,
        ),
        category=PaperCategory.RECENT,
        rationale="Test paper.",
    )


@pytest.fixture()
def mock_chroma():
    chroma = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 0
    chroma.get_or_create_collection.return_value = collection
    chroma.get_collection.return_value = collection
    return chroma, collection


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 1024]
    embedder.embed_single.return_value = [0.1] * 1024
    return embedder


def test_embed_and_store_calls_upsert(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    papers = [_make_paper("p1", "This is a valid abstract with enough content to embed properly.")]
    result = store.embed_and_store("session-123", papers)
    assert result["stored"] >= 1
    collection.upsert.assert_called()


def test_short_abstract_flagged_not_embeddable(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    papers = [_make_paper("p2", "Too short")]
    result = store.embed_and_store("session-short", papers)
    assert result["skipped_short"] == 1
    call_kwargs = collection.upsert.call_args[1]
    assert call_kwargs["metadatas"][0]["embeddable"] is False


def test_existing_collection_reused(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    collection.count.return_value = 5
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    papers = [_make_paper("p3", "Some abstract content here that is long enough to embed.")]
    result = store.embed_and_store("session-existing", papers, skip_if_exists=True)
    assert result.get("reused") == 1
    collection.upsert.assert_not_called()


def test_embedding_error_skips_chunk_does_not_halt(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    mock_embedder.embed_batch.return_value = [None]
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    papers = [_make_paper("p4", "This abstract is long enough to be embedded properly.")]
    result = store.embed_and_store("session-err", papers)
    assert result["skipped_error"] >= 1
    collection.upsert.assert_not_called()


def test_similarity_search_returns_ranked_results(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    collection.count.return_value = 3
    collection.query.return_value = {
        "documents": [["chunk text one", "chunk text two", "chunk text three"]],
        "metadatas": [
            [
                {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "year": "2024",
                    "source": "arxiv",
                    "classification": "recent",
                    "chunk_index": 0,
                    "embeddable": True,
                },
                {
                    "paper_id": "p2",
                    "title": "Paper 2",
                    "year": "2023",
                    "source": "semantic_scholar",
                    "classification": "foundational",
                    "chunk_index": 0,
                    "embeddable": True,
                },
                {
                    "paper_id": "p3",
                    "title": "Paper 3",
                    "year": "2019",
                    "source": "arxiv",
                    "classification": "seminal",
                    "chunk_index": 1,
                    "embeddable": True,
                },
            ]
        ],
        "distances": [[0.05, 0.15, 0.30]],
    }
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    results = store.similarity_search("session-123", "transformer attention", top_k=3)
    assert len(results) == 3
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    assert results[0]["paper_id"] == "p1"


def test_similarity_search_empty_when_no_collection(mock_chroma, mock_embedder) -> None:
    chroma, collection = mock_chroma
    collection.count.return_value = 0
    store = VectorStore(chroma_client=chroma, embedding_client=mock_embedder)
    results = store.similarity_search("session-none", "any query", top_k=5)
    assert results == []
