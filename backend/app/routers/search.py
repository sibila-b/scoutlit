from __future__ import annotations

import logging

import chromadb
from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.models.search import ChunkResult, SimilaritySearchResponse
from backend.app.services.embedding import EmbeddingClient
from backend.app.services.vector_store import ChunkResult as _SvcChunkResult
from backend.app.services.vector_store import search_similar

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _to_response_model(r: _SvcChunkResult) -> ChunkResult:
    return ChunkResult(
        chunk_text=r.chunk_text,
        chunk_index=r.chunk_index,
        distance=r.distance,
        paper_id=r.paper_id,
        title=r.title,
        year=r.year,
        citation_count=r.citation_count,
        source=r.source,
        classification=r.classification,
        embeddable=r.embeddable,
    )


@router.get("/search/similar", response_model=SimilaritySearchResponse)
def get_similar(
    request: Request,
    query: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    session_id: str = Query(..., description="Session UUID identifying the ChromaDB collection"),
) -> SimilaritySearchResponse:
    chroma: chromadb.HttpClient | None = request.app.state.chroma
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB is not available.")

    try:
        embedder = EmbeddingClient()
        chunks = search_similar(
            session_id=session_id,
            query=query,
            top_k=top_k,
            chroma=chroma,
            embedder=embedder,
        )
    except Exception as exc:
        logger.exception("Search failed for session %s", session_id)
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    if not chunks and chroma is not None:
        try:
            chroma.get_collection(name=session_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return SimilaritySearchResponse(
        results=[_to_response_model(c) for c in chunks],
        total=len(chunks),
        session_id=session_id,
    )
