from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100


class EmbeddingClient:
    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a list of texts in batches. Returns None for any chunk whose batch fails."""
        results: list[list[float] | None] = [None] * len(texts)
        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, len(texts))
            batch = texts[batch_start:batch_end]
            try:
                response = self._client.embeddings.create(model=_MODEL, input=batch)
                for i, obj in enumerate(response.data):
                    results[batch_start + i] = obj.embedding
            except Exception as exc:
                logger.warning(
                    "Embedding batch [%d:%d] failed, skipping: %s", batch_start, batch_end, exc
                )
        return results

    def embed_query(self, query: str) -> list[float]:
        response = self._client.embeddings.create(model=_MODEL, input=[query])
        return response.data[0].embedding
