from __future__ import annotations

import asyncio
import logging
import os

import httpx

from backend.app.models.paper_search import PaperResult

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,authors,abstract,year,externalIds,openAccessPdf,citationCount"
_MAX_RETRIES = 3


async def fetch_semantic_scholar(
    query: str, max_results: int
) -> tuple[list[PaperResult], list[str]]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}

    try:
        async with httpx.AsyncClient(base_url=_BASE_URL, headers=headers, timeout=30) as client:
            resp = None
            for attempt in range(_MAX_RETRIES):
                resp = await client.get(
                    "/paper/search",
                    params={"query": query, "limit": max_results, "fields": _FIELDS},
                )
                if resp.status_code != 429:
                    break
                wait = 2 ** (attempt + 1)
                logger.warning("Semantic Scholar 429 — waiting %ds (attempt %d)", wait, attempt + 1)
                await asyncio.sleep(wait)

            if resp is None or resp.status_code == 429:
                return [], ["Semantic Scholar rate-limited after 3 attempts."]

            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Semantic Scholar fetch failed: %s", exc)
        return [], [f"Semantic Scholar unavailable: {exc}"]

    papers: list[PaperResult] = []
    for item in resp.json().get("data", []):
        pdf = item.get("openAccessPdf") or {}
        year = str(item.get("year", "")) if item.get("year") else ""
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI")
        papers.append(
            PaperResult(
                id=item["paperId"],
                title=item.get("title", ""),
                authors=[a["name"] for a in item.get("authors", [])],
                abstract=item.get("abstract") or "",
                year=year,
                citation_count=item.get("citationCount"),
                source="semantic_scholar",
                url=pdf.get("url") or f"https://www.semanticscholar.org/paper/{item['paperId']}",
                doi=doi,
            )
        )
    return papers, []
