from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("chromadb.HttpClient") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            from backend.app.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


def test_undefined_route_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/undefined-route")
    assert response.status_code == 404


def test_undefined_route_structured_body(client: TestClient) -> None:
    data = client.get("/api/v1/undefined-route").json()
    assert "error" in data
    assert "message" in data
    assert "status" in data
    assert data["status"] == 404


def test_undefined_route_no_stacktrace(client: TestClient) -> None:
    text = client.get("/api/v1/undefined-route").text
    assert "Traceback" not in text
    assert "traceback" not in text


def test_unhandled_exception_returns_500(client: TestClient) -> None:
    # Inject a route that raises an unhandled exception
    from backend.app.main import app as _app

    @_app.get("/api/v1/boom")
    def boom():  # type: ignore[return]
        raise RuntimeError("something went wrong")

    response = client.get("/api/v1/boom")
    assert response.status_code == 500


def test_unhandled_exception_no_stacktrace(client: TestClient) -> None:
    text = client.get("/api/v1/boom").text
    assert "Traceback" not in text
    assert "something went wrong" not in text


def test_500_structured_body(client: TestClient) -> None:
    data = client.get("/api/v1/boom").json()
    assert data["error"] == "InternalServerError"
    assert "message" in data
    assert data["status"] == 500
