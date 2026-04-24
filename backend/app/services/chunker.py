from __future__ import annotations

from dataclasses import dataclass

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 512
_OVERLAP_TOKENS = 64
_MIN_ABSTRACT_CHARS = 20


@dataclass
class Chunk:
    text: str
    chunk_index: int
    embeddable: bool


def chunk_abstract(abstract: str) -> list[Chunk]:
    if len(abstract) < _MIN_ABSTRACT_CHARS:
        return [Chunk(text=abstract, chunk_index=0, embeddable=False)]

    tokens = _ENCODING.encode(abstract)

    if len(tokens) <= _MAX_TOKENS:
        return [Chunk(text=abstract, chunk_index=0, embeddable=True)]

    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + _MAX_TOKENS, len(tokens))
        text = _ENCODING.decode(tokens[start:end])
        chunks.append(Chunk(text=text, chunk_index=len(chunks), embeddable=True))
        if end == len(tokens):
            break
        start = end - _OVERLAP_TOKENS

    return chunks
