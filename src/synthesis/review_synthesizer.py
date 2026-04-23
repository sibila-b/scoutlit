from __future__ import annotations

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


class ReviewSynthesizer:
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def synthesize(self, topic: str, papers: list[ClassifiedPaper]) -> LiteratureReview:
        paper_summaries = "\n\n".join(
            f"[{p.paper.authors[0].split()[-1] if p.paper.authors else 'Unknown'} et al., "
            f"{p.paper.published[:4]}] ({p.category})\n"
            f"Title: {p.paper.title}\n"
            f"Abstract: {p.paper.abstract[:400]}"
            for p in papers
        )
        prompt = f"Research topic: {topic}\n\nPapers:\n{paper_summaries}"

        with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            full_text = stream.get_final_text()

        sections = self._parse_sections(full_text)
        gaps = self._extract_gaps(sections.get("Research Gaps", ""))

        return LiteratureReview(
            topic=topic,
            summary=full_text,
            sections=sections,
            gaps=gaps,
        )

    def _parse_sections(self, text: str) -> dict[str, str]:
        known = ["Background", "Key Contributions", "Recent Advances", "Research Gaps"]
        sections: dict[str, str] = {}
        for i, heading in enumerate(known):
            start = text.find(heading)
            if start == -1:
                continue
            end = len(text)
            for other in known[i + 1 :]:
                pos = text.find(other, start)
                if pos != -1:
                    end = min(end, pos)
            sections[heading] = text[start + len(heading) : end].strip(" :\n")
        return sections

    def _extract_gaps(self, gaps_text: str) -> list[str]:
        return [line.lstrip("-•* ") for line in gaps_text.splitlines() if line.strip()]
