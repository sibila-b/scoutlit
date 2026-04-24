from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    chunk_text: str
    chunk_index: int
    distance: float
    paper_id: str
    title: str
    year: str
    citation_count: int
    source: str
    classification: str
    embeddable: bool


class SimilaritySearchResponse(BaseModel):
    results: list[ChunkResult]
    total: int
    session_id: str
