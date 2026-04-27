from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.app.models.paper_search import SearchRequest, SearchResponse
from backend.app.services.search_orchestrator import run_search

logger = logging.getLogger(__name__)
router = APIRouter(tags=["papers"])

_MAX_RESULTS_HARD_CAP = 20


@router.post("/search", response_model=SearchResponse)
async def search_papers(body: SearchRequest) -> SearchResponse:
    warnings: list[str] = []
    max_results = body.max_results

    if max_results > _MAX_RESULTS_HARD_CAP:
        warnings.append(
            f"max_results clamped to {_MAX_RESULTS_HARD_CAP} (requested {max_results})."
        )
        max_results = _MAX_RESULTS_HARD_CAP

    papers, source_warnings = await run_search(body.topic, body.sources, max_results)
    warnings.extend(source_warnings)

    return SearchResponse(
        papers=papers,
        total=len(papers),
        warnings=warnings,
        message=f"Found {len(papers)} papers for '{body.topic}'.",
    )
