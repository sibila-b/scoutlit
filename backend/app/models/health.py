from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    status: int
