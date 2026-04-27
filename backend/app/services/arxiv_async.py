from __future__ import annotations

import logging

import feedparser
import httpx

from backend.app.models.paper_search import PaperResult

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"


async def fetch_arxiv(query: str, max_results: int) -> tuple[list[PaperResult], list[str]]:
    params = {"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(_ARXIV_API, params=params)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("ArXiv fetch failed: %s", exc)
        return [], [f"ArXiv unavailable: {exc}"]

    feed = feedparser.parse(resp.text)
    papers: list[PaperResult] = []
    for entry in feed.entries:
        paper_id = entry.id.split("/abs/")[-1]
        published = entry.get("published", "")
        year = published[:4] if published else ""
        papers.append(
            PaperResult(
                id=paper_id,
                title=entry.get("title", "").replace("\n", " ").strip(),
                authors=[a.get("name", "") for a in entry.get("authors", [])],
                abstract=entry.get("summary", "").replace("\n", " ").strip(),
                year=year,
                citation_count=None,
                source="arxiv",
                url=entry.id,
                doi=None,
            )
        )
    return papers, []
