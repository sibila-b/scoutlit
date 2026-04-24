from __future__ import annotations

import os

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .arxiv_client import Paper

_BASE_URL = "https://api.semanticscholar.org/graph/v1"

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


_FIELDS = "paperId,title,authors,abstract,year,externalIds,openAccessPdf,citationCount"


class SemanticScholarClient:
    def __init__(self) -> None:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"x-api-key": api_key} if api_key else {}
        self._http = httpx.Client(base_url=_BASE_URL, headers=headers, timeout=30)

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def search(self, query: str, limit: int = 50) -> list[Paper]:
        resp = self._http.get(
            "/paper/search",
            params={"query": query, "limit": limit, "fields": _FIELDS},
        )
        resp.raise_for_status()
        papers = []
        for item in resp.json().get("data", []):
            pdf = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    id=item["paperId"],
                    title=item.get("title", ""),
                    authors=[a["name"] for a in item.get("authors", [])],
                    abstract=item.get("abstract") or "",
                    published=str(item.get("year", "")),
                    url=pdf.get("url", f"https://www.semanticscholar.org/paper/{item['paperId']}"),
                    source="semantic_scholar",
                    citation_count=item.get("citationCount"),
                )
            )
        return papers
