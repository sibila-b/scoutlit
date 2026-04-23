from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from backend.app import VERSION
from backend.app.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=VERSION,
        timestamp=datetime.now(UTC).isoformat(),
    )
