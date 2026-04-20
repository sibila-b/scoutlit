from __future__ import annotations

import arxiv
from pydantic import BaseModel


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    url: str
    source: str = "arxiv"


class ArxivClient:
    def __init__(self, max_results: int = 50) -> None:
        self._max_results = max_results
        self._client = arxiv.Client()

    def search(self, query: str) -> list[Paper]:
        search = arxiv.Search(query=query, max_results=self._max_results)
        results = []
        for r in self._client.results(search):
            results.append(
                Paper(
                    id=r.entry_id,
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    abstract=r.summary,
                    published=r.published.isoformat(),
                    url=r.entry_id,
                )
            )
        return results
