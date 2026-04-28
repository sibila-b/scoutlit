from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.models.paper import PaperResult as PaperResult  # re-export


class SearchRequest(BaseModel):
    topic: str
    sources: list[str] = ["arxiv", "semantic_scholar"]
    max_results: int = Field(default=10, ge=1)

    @field_validator("topic")
    @classmethod
    def topic_min_length(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("topic must be at least 3 characters")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        valid = {"arxiv", "semantic_scholar"}
        if not v:
            raise ValueError("At least one source must be specified.")
        invalid = [s for s in v if s not in valid]
        if invalid:
            raise ValueError(f"Unknown sources: {invalid}. Valid: {sorted(valid)}")
        return v


class SearchResponse(BaseModel):
    papers: list[PaperResult]
    total: int
    warnings: list[str]
    message: str
