from __future__ import annotations

import pytest

from backend.app.models.paper_search import PaperResult, SearchRequest
from backend.app.services.paper_deduplicator import deduplicate, sort_by_relevance


def _paper(
    paper_id: str,
    title: str,
    source: str = "arxiv",
    doi: str | None = None,
    citation_count: int | None = None,
) -> PaperResult:
    return PaperResult(
        id=paper_id,
        title=title,
        authors=["Author A"],
        abstract="Abstract.",
        year="2024",
        citation_count=citation_count,
        source=source,
        url=f"https://example.com/{paper_id}",
        doi=doi,
    )


# --- deduplication ---


def test_deduplicate_same_doi_removes_duplicate() -> None:
    papers = [
        _paper("a1", "Attention Is All You Need", doi="10.1234/xyz"),
        _paper("b1", "Attention Is All You Need", source="semantic_scholar", doi="10.1234/xyz"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0].id == "a1"


def test_deduplicate_different_doi_both_kept() -> None:
    papers = [
        _paper("a1", "Paper A", doi="10.1234/aaa"),
        _paper("b1", "Paper B", doi="10.1234/bbb"),
    ]
    result = deduplicate(papers)
    assert len(result) == 2


def test_deduplicate_same_normalized_title_removes_duplicate() -> None:
    papers = [
        _paper("a1", "BERT: Pre-training of Deep Bidirectional Transformers"),
        _paper("b1", "bert  pre training of deep bidirectional transformers"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1


def test_deduplicate_different_title_both_kept() -> None:
    papers = [
        _paper("a1", "Paper Alpha"),
        _paper("b1", "Paper Beta"),
    ]
    result = deduplicate(papers)
    assert len(result) == 2


def test_deduplicate_no_doi_uses_title() -> None:
    papers = [
        _paper("a1", "Transformers Are Great"),
        _paper("b1", "Transformers Are Great", source="semantic_scholar"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1


def test_deduplicate_empty_list() -> None:
    assert deduplicate([]) == []


# --- sort by relevance ---


def test_sort_by_relevance_descending() -> None:
    papers = [
        _paper("a1", "Low cited", citation_count=5),
        _paper("b1", "High cited", citation_count=1000),
        _paper("c1", "Mid cited", citation_count=100),
    ]
    result = sort_by_relevance(papers)
    assert [p.id for p in result] == ["b1", "c1", "a1"]


def test_sort_by_relevance_none_treated_as_zero() -> None:
    papers = [
        _paper("a1", "No citations", citation_count=None),
        _paper("b1", "Some citations", citation_count=1),
    ]
    result = sort_by_relevance(papers)
    assert result[0].id == "b1"


# --- input validation ---


def test_search_request_topic_too_short() -> None:
    with pytest.raises(Exception):
        SearchRequest(topic="ab")


def test_search_request_topic_strips_whitespace() -> None:
    req = SearchRequest(topic="  transformers  ")
    assert req.topic == "transformers"


def test_search_request_invalid_source_raises() -> None:
    with pytest.raises(Exception):
        SearchRequest(topic="transformers", sources=["invalid_source"])


def test_search_request_empty_sources_raises() -> None:
    with pytest.raises(Exception):
        SearchRequest(topic="transformers", sources=[])


def test_search_request_defaults() -> None:
    req = SearchRequest(topic="machine learning")
    assert set(req.sources) == {"arxiv", "semantic_scholar"}
    assert req.max_results == 10
