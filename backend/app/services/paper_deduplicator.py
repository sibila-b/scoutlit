from __future__ import annotations

import re
import unicodedata

from src.models.paper import PaperResult


def _normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title)
    title = title.lower()
    title = re.sub(r"[-/]", " ", title)  # hyphens/slashes → spaces before stripping punct
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def deduplicate(papers: list[PaperResult]) -> list[PaperResult]:
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    result: list[PaperResult] = []

    for paper in papers:
        if paper.doi:
            if paper.doi in seen_dois:
                continue
            seen_dois.add(paper.doi)

        norm = _normalize_title(paper.title)
        if norm in seen_titles:
            continue
        seen_titles.add(norm)

        result.append(paper)

    return result


def sort_by_relevance(papers: list[PaperResult]) -> list[PaperResult]:
    return sorted(papers, key=lambda p: p.citation_count or 0, reverse=True)
