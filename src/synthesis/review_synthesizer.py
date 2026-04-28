from __future__ import annotations

import re

import anthropic
from pydantic import BaseModel

from src.classification.paper_classifier import ClassifiedPaper


class LiteratureReview(BaseModel):
    topic: str
    summary: str
    sections: dict[str, str]
    gaps: list[str]


_SYSTEM = """\
You are an expert academic writer. Given classified papers, produce a structured literature review
with inline citations in the format [Author et al., Year]. Include sections for:
Background, Key Contributions, Recent Advances, and Research Gaps.
"""

_SECTION_HEADINGS = ["Background", "Key Contributions", "Recent Advances", "Research Gaps"]


def _heading_pattern(heading: str) -> re.Pattern[str]:
    h = re.escape(heading)
    # Handles: ## Heading | **Heading** | *Heading* | Heading: at line/string start
    return re.compile(
        r"#{1,3}\s*" + h + r"|\*{1,2}" + h + r"\*{1,2}|^" + h + r":?",
        re.IGNORECASE | re.MULTILINE,
    )


def _truncate_at_word(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


class ReviewSynthesizer:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def synthesize(self, topic: str, papers: list[ClassifiedPaper]) -> LiteratureReview:
        paper_summaries = "\n\n".join(
            # NOTE: takes last word as surname — fails for compound surnames (van den Berg)
            # and single-name authors; acceptable for citation format at this stage.
            f"[{p.paper.authors[0].split()[-1] if p.paper.authors else 'Unknown'} et al., "
            f"{p.paper.year}] ({p.category})\n"
            f"Title: {p.paper.title}\n"
            f"Abstract: {_truncate_at_word(p.paper.abstract)}"
            for p in papers
        )
        prompt = f"Research topic: {topic}\n\nPapers:\n{paper_summaries}"

        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                full_text = stream.get_final_text()
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error during synthesis: {exc}") from exc

        sections = self._parse_sections(full_text)
        # TODO SL-08: replace with dedicated gap detection call (Sprint 2)
        gaps = self._extract_gaps(sections.get("Research Gaps", ""))

        return LiteratureReview(
            topic=topic,
            summary=full_text,
            sections=sections,
            gaps=gaps,
        )

    def _parse_sections(self, text: str) -> dict[str, str]:
        positions: list[tuple[int, str, int]] = []  # (match_start, heading, content_start)
        for heading in _SECTION_HEADINGS:
            match = _heading_pattern(heading).search(text)
            if match:
                positions.append((match.start(), heading, match.end()))
        positions.sort()

        sections: dict[str, str] = {}
        for i, (_, heading, content_start) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections[heading] = text[content_start:end].strip(" :\n")
        return sections

    def _extract_gaps(self, gaps_text: str) -> list[str]:
        return [line.lstrip("-•* ") for line in gaps_text.splitlines() if line.strip()]
