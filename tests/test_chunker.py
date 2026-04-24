from __future__ import annotations

import tiktoken

from src.embedding.chunker import chunk_text

_ENC = tiktoken.get_encoding("cl100k_base")


def test_empty_string_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk() -> None:
    text = "This is a short abstract."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_within_limit_returns_single_chunk() -> None:
    tokens = _ENC.encode("word ") * 102  # ~510 tokens
    text = _ENC.decode(tokens)
    chunks = chunk_text(text, max_tokens=512, overlap=64)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks() -> None:
    tokens = _ENC.encode("abstract word ") * 260  # ~520 tokens after roundtrip
    text = _ENC.decode(tokens)
    chunks = chunk_text(text, max_tokens=512, overlap=64)
    assert len(chunks) == 2


def test_no_chunk_exceeds_max_tokens() -> None:
    long_text = " ".join(["word"] * 2000)
    chunks = chunk_text(long_text, max_tokens=512, overlap=64)
    for chunk in chunks:
        assert len(_ENC.encode(chunk)) <= 512


def test_chunks_overlap_correctly() -> None:
    tokens = _ENC.encode("token ") * 600
    text = _ENC.decode(tokens)
    chunks = chunk_text(text, max_tokens=512, overlap=64)
    chunk1_tokens = _ENC.encode(chunks[0])
    chunk2_tokens = _ENC.encode(chunks[1])
    assert chunk1_tokens[-64:] == chunk2_tokens[:64]


def test_single_chunk_when_exactly_at_max() -> None:
    tokens = _ENC.encode("x ") * 256  # 512 tokens
    text = _ENC.decode(tokens)
    chunks = chunk_text(text, max_tokens=512, overlap=64)
    assert len(chunks) == 1
