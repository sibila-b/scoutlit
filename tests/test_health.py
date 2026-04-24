from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "VOYAGE_API_KEY": "test-key"}):
        with patch("chromadb.HttpClient") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            from backend.app.main import app

            with TestClient(app) as c:
                yield c


def test_health_status_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_schema(client: TestClient) -> None:
    data = client.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_health_content_type(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert "application/json" in response.headers["content-type"]


def test_health_responds_under_100ms(client: TestClient) -> None:
    start = time.perf_counter()
    client.get("/api/v1/health")
    assert (time.perf_counter() - start) * 1000 < 100


def test_openapi_docs_accessible(client: TestClient) -> None:
    response = client.get("/api/v1/docs")
    assert response.status_code == 200


def test_openapi_json_accessible(client: TestClient) -> None:
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "0.1.0"
