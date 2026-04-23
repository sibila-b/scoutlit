from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY"]


def _validate_env() -> None:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variable(s): {', '.join(missing)}")


_validate_env()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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


app = FastAPI(title="ScoutLit API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
