from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.embedding.store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


class SimilarityRequest(BaseModel):
    session_id: str = Field(..., description="UUID identifying the search session corpus")
    query: str = Field(..., min_length=3, description="Query string to search against")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")


class ChunkResult(BaseModel):
    text: str
    score: float
    paper_id: str
    title: str
    year: str
    source: str
    classification: str
    chunk_index: int
    embeddable: bool


class SimilarityResponse(BaseModel):
    session_id: str
    query: str
    results: list[ChunkResult]


@router.post("/search/similar", response_model=SimilarityResponse)
def search_similar(body: SimilarityRequest, request: Request) -> SimilarityResponse:
    chroma = request.app.state.chroma
    if chroma is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    store = VectorStore(chroma_client=chroma)

    try:
        chunks = store.similarity_search(
            session_id=body.session_id,
            query=body.query,
            top_k=body.top_k,
        )
    except RuntimeError as exc:
        logger.error("Similarity search failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SimilarityResponse(
        session_id=body.session_id,
        query=body.query,
        results=[ChunkResult(**c) for c in chunks],
    )
