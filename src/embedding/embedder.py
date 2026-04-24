from __future__ import annotations

import logging
import os

import voyageai

logger = logging.getLogger(__name__)

_MODEL = "voyage-3"
_BATCH_SIZE = 128


class EmbeddingClient:
    def __init__(self, client: voyageai.Client | None = None) -> None:
        self._client = client or voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed texts in batches. Returns None in place of any failed batch's items."""
        if not texts:
            return []

        all_embeddings: list[list[float] | None] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                result = self._client.embed(batch, model=_MODEL, input_type="document")
                all_embeddings.extend(result.embeddings)
            except Exception as exc:
                logger.warning(
                    "Embedding batch %d failed: %s — skipping %d texts",
                    i // _BATCH_SIZE,
                    exc,
                    len(batch),
                )
                all_embeddings.extend([None] * len(batch))

        return all_embeddings

    def embed_single(self, text: str) -> list[float] | None:
        """Embed a single text. Returns None on failure."""
        results = self.embed_batch([text])
        return results[0] if results else None
