from __future__ import annotations

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
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def detect(self, topic: str, papers: list[ClassifiedPaper]) -> list[ResearchGap]:
        import json

        paper_list = "\n".join(
            f"- {p.paper.title} ({p.paper.published[:4]}, {p.category})" for p in papers
        )
        prompt = f"Topic: {topic}\n\nPapers:\n{paper_list}"

        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = json.loads(message.content[0].text)
        return [ResearchGap(**item) for item in raw]
