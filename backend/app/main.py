from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app import VERSION
from backend.app.routers import health, paper_search, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"]


def _validate_env() -> None:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variable(s): {', '.join(missing)}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _validate_env()
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        logger.info("Connected to ChromaDB at %s:%s", host, port)
        app.state.chroma = client
    except Exception as exc:
        logger.error("Failed to connect to ChromaDB at %s:%s — %s", host, port, exc)
        app.state.chroma = None
    yield


app = FastAPI(
    title="ScoutLit API",
    version=VERSION,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

_cors_origins = [
    u.strip() for u in os.getenv("FRONTEND_URL", "http://localhost:3000").split(",") if u.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %s %.1fms", request.method, request.url.path, response.status_code, duration_ms
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc.detail),
            "status": exc.status_code,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Request validation failed.",
            "status": 422,
            "detail": json.loads(json.dumps(exc.errors(), default=str)),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "status": 500,
        },
    )


app.include_router(health.router, prefix="/api/v1")
app.include_router(paper_search.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
