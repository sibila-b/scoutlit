from __future__ import annotations

import arxiv

from src.models.paper import PaperResult


class ArxivClient:
    def __init__(self, max_results: int = 50) -> None:
        self._max_results = max_results
        self._client = arxiv.Client()

    def search(self, query: str) -> list[PaperResult]:
        search = arxiv.Search(query=query, max_results=self._max_results)
        results = []
        try:
            for r in self._client.results(search):
                results.append(
                    PaperResult(
                        id=r.get_short_id(),
                        title=r.title,
                        authors=[a.name for a in r.authors],
                        abstract=r.summary,
                        year=r.published.isoformat()[:4],
                        url=r.entry_id,
                    )
                )
        except arxiv.ArxivError as exc:
            raise RuntimeError(f"arXiv search failed: {exc}") from exc
        return results
