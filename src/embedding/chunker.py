from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 512
_OVERLAP_TOKENS = 64


def chunk_text(
    text: str,
    max_tokens: int = _MAX_TOKENS,
    overlap: int = _OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping chunks of max_tokens with overlap tokens between chunks."""
    if not text.strip():
        return []

    tokens = _ENCODING.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap

    return chunks
