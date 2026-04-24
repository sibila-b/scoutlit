from __future__ import annotations

import json
from enum import StrEnum

import anthropic
from pydantic import BaseModel

from src.retrieval.arxiv_client import Paper


class PaperCategory(StrEnum):
    SEMINAL = "seminal"
    FOUNDATIONAL = "foundational"
    RECENT = "recent"


class ClassifiedPaper(BaseModel):
    paper: Paper
    category: PaperCategory
    rationale: str


_SYSTEM = """\
You are a research librarian. Given a paper's metadata and the user's research topic,
classify the paper as one of:
- seminal: highly-cited work that introduced a key concept
- foundational: important background/survey work
- recent: recent contribution (typically last 3 years)

Respond with valid JSON: {"category": "<value>", "rationale": "<one sentence>"}
"""


def _truncate_at_word(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


class PaperClassifier:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model

    def classify(self, paper: Paper, topic: str) -> ClassifiedPaper:
        prompt = (
            f"Topic: {topic}\n\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:3])}\n"
            f"Year: {paper.published[:4]}\n"
            f"Abstract: {_truncate_at_word(paper.abstract)}"
        )
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error during classification: {exc}") from exc
        data = json.loads(message.content[0].text)
        return ClassifiedPaper(
            paper=paper,
            category=PaperCategory(data["category"]),
            rationale=data["rationale"],
        )

    def classify_batch(self, papers: list[Paper], topic: str) -> list[ClassifiedPaper]:
        return [self.classify(p, topic) for p in papers]
