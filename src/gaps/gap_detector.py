from __future__ import annotations

import json

import anthropic
from pydantic import BaseModel

from src.classification.paper_classifier import ClassifiedPaper


class ResearchGap(BaseModel):
    description: str
    suggested_question: str
    supporting_papers: list[str]


_SYSTEM = """\
You are a research strategist. Given a set of classified papers on a topic, identify concrete
research gaps — areas not yet studied, contradictions between findings, or promising extensions.
For each gap output JSON with keys: description, suggested_question, supporting_papers (list of
titles).
Return a JSON array.
"""


class GapDetector:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model

    def detect(self, topic: str, papers: list[ClassifiedPaper]) -> list[ResearchGap]:
        paper_list = "\n".join(f"- {p.paper.title} ({p.paper.year}, {p.category})" for p in papers)
        prompt = f"Topic: {topic}\n\nPapers:\n{paper_list}"

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error during gap detection: {exc}") from exc
        raw = json.loads(message.content[0].text)
        return [ResearchGap(**item) for item in raw]
