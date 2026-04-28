from __future__ import annotations

from pydantic import BaseModel


class PaperResult(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    year: str
    url: str
    source: str = "arxiv"
    citation_count: int | None = None
    doi: str | None = None
