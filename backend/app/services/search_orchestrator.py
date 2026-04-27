from __future__ import annotations

import asyncio
import logging

from backend.app.models.paper_search import PaperResult
from backend.app.services.arxiv_async import fetch_arxiv
from backend.app.services.paper_deduplicator import deduplicate, sort_by_relevance
from backend.app.services.semantic_scholar_async import fetch_semantic_scholar

logger = logging.getLogger(__name__)


async def run_search(
    topic: str,
    sources: list[str],
    max_results: int,
) -> tuple[list[PaperResult], list[str]]:
    tasks = []
    if "arxiv" in sources:
        tasks.append(fetch_arxiv(topic, max_results))
    if "semantic_scholar" in sources:
        tasks.append(fetch_semantic_scholar(topic, max_results))

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[PaperResult] = []
    warnings: list[str] = []
    for item in raw:
        if isinstance(item, BaseException):
            logger.warning("Source raised exception: %s", item)
            warnings.append(f"Source error: {item}")
        else:
            papers, warns = item
            all_papers.extend(papers)
            warnings.extend(warns)

    deduped = deduplicate(all_papers)
    sorted_papers = sort_by_relevance(deduped)
    return sorted_papers[:max_results], warnings
