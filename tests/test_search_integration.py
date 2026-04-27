from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.models.paper_search import PaperResult


@pytest.fixture(scope="module")
def client():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "VOYAGE_API_KEY": "test-key"}):
        with patch("chromadb.HttpClient") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            from backend.app.main import app

            with TestClient(app) as c:
                yield c


def _mock_paper(paper_id: str = "p1", title: str = "Test Paper") -> PaperResult:
    return PaperResult(
        id=paper_id,
        title=title,
        authors=["Author A"],
        abstract="Some abstract.",
        year="2024",
        citation_count=42,
        source="arxiv",
        url=f"https://arxiv.org/abs/{paper_id}",
        doi=None,
    )


def test_search_returns_200(client: TestClient) -> None:
    with patch(
        "backend.app.routers.paper_search.run_search",
        new=AsyncMock(return_value=([_mock_paper()], [])),
    ):
        response = client.post("/api/v1/search", json={"topic": "attention mechanism"})
    assert response.status_code == 200


def test_search_response_schema(client: TestClient) -> None:
    with patch(
        "backend.app.routers.paper_search.run_search",
        new=AsyncMock(return_value=([_mock_paper()], [])),
    ):
        data = client.post("/api/v1/search", json={"topic": "attention mechanism"}).json()
    assert "papers" in data
    assert "total" in data
    assert "warnings" in data
    assert "message" in data
    assert data["total"] == 1
    assert data["papers"][0]["title"] == "Test Paper"


def test_search_short_topic_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"topic": "ab"})
    assert response.status_code == 422


def test_search_invalid_source_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/search", json={"topic": "machine learning", "sources": ["invalid"]}
    )
    assert response.status_code == 422


def test_search_empty_sources_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/search", json={"topic": "neural networks", "sources": []})
    assert response.status_code == 422


def test_search_max_results_clamped(client: TestClient) -> None:
    with patch(
        "backend.app.routers.paper_search.run_search",
        new=AsyncMock(return_value=([], [])),
    ):
        data = client.post(
            "/api/v1/search",
            json={"topic": "neural networks", "max_results": 99},
        ).json()
    assert any("clamped" in w for w in data["warnings"])


def test_search_multiple_papers_returned(client: TestClient) -> None:
    papers = [_mock_paper("a1", "Paper A"), _mock_paper("b1", "Paper B")]
    with patch(
        "backend.app.routers.paper_search.run_search",
        new=AsyncMock(return_value=(papers, [])),
    ):
        data = client.post(
            "/api/v1/search",
            json={"topic": "transformers", "sources": ["arxiv", "semantic_scholar"]},
        ).json()
    assert data["total"] == 2


def test_search_source_warning_propagated(client: TestClient) -> None:
    with patch(
        "backend.app.routers.paper_search.run_search",
        new=AsyncMock(return_value=([], ["ArXiv unavailable: timeout"])),
    ):
        data = client.post("/api/v1/search", json={"topic": "quantum computing"}).json()
    assert any("ArXiv" in w for w in data["warnings"])
