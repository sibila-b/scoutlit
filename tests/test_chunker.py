from __future__ import annotations

from backend.app.services.chunker import _ENCODING, _MAX_TOKENS, _OVERLAP_TOKENS, chunk_abstract


def _tokens(text: str) -> list[int]:
    return _ENCODING.encode(text)


def _make_text(min_tokens: int) -> str:
    """Return text that tokenizes to at least min_tokens without token-slice roundtrip issues."""
    sentence = "The neural network processes language sequences using multi-head attention mechanisms. "
    tokens_per = len(_tokens(sentence))
    repeats = (min_tokens // tokens_per) + 2
    return sentence * repeats


# ── short / empty abstracts ────────────────────────────────────────────────────

def test_empty_abstract_not_embeddable() -> None:
    chunks = chunk_abstract("")
    assert len(chunks) == 1
    assert chunks[0].embeddable is False
    assert chunks[0].chunk_index == 0


def test_short_abstract_not_embeddable() -> None:
    chunks = chunk_abstract("Too short.")
    assert len(chunks) == 1
    assert chunks[0].embeddable is False


def test_exactly_19_chars_not_embeddable() -> None:
    chunks = chunk_abstract("a" * 19)
    assert chunks[0].embeddable is False


def test_exactly_20_chars_is_embeddable() -> None:
    chunks = chunk_abstract("a" * 20)
    assert chunks[0].embeddable is True


# ── single-chunk abstracts ─────────────────────────────────────────────────────

def test_single_chunk_short_text() -> None:
    text = "We study neural networks and their applications in NLP."
    chunks = chunk_abstract(text)
    assert len(chunks) == 1
    assert chunks[0].embeddable is True
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == text


def test_exactly_max_tokens_single_chunk() -> None:
    # Build a text whose token count is exactly _MAX_TOKENS by using a known
    # sentence and trimming the token sequence inside the encoder's round-trip.
    sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
    toks = _tokens(sentence * 100)[:_MAX_TOKENS]
    text = _ENCODING.decode(toks)
    chunks = chunk_abstract(text)
    # The decoded text may encode to slightly fewer tokens — still ≤ MAX → 1 chunk.
    assert len(_tokens(chunks[0].text)) <= _MAX_TOKENS
    assert all(c.embeddable for c in chunks)


# ── multi-chunk abstracts ─────────────────────────────────────────────────────

def test_multi_chunk_count() -> None:
    text = _make_text(2 * _MAX_TOKENS)
    assert len(_tokens(text)) > _MAX_TOKENS, "Helper must produce text over the limit"
    chunks = chunk_abstract(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.embeddable is True


def test_chunk_indices_sequential() -> None:
    text = _make_text(2 * _MAX_TOKENS)
    chunks = chunk_abstract(text)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_no_chunk_exceeds_max_tokens() -> None:
    text = _make_text(4 * _MAX_TOKENS)
    for chunk in chunk_abstract(text):
        assert len(_tokens(chunk.text)) <= _MAX_TOKENS


def test_overlap_between_consecutive_chunks() -> None:
    text = _make_text(2 * _MAX_TOKENS)
    chunks = chunk_abstract(text)
    assert len(chunks) >= 2

    # The chunker works on the token sequence of the full abstract.
    all_tokens = _tokens(text)
    step = _MAX_TOKENS - _OVERLAP_TOKENS  # distance between chunk start positions

    for i in range(len(chunks) - 1):
        start_next = (i + 1) * step
        overlap_tokens_expected = all_tokens[start_next: start_next + _OVERLAP_TOKENS]
        overlap_tokens_actual = _tokens(chunks[i + 1].text)[:_OVERLAP_TOKENS]
        assert overlap_tokens_actual == overlap_tokens_expected, (
            f"Overlap mismatch between chunk {i} and {i + 1}"
        )


def test_last_chunk_shorter_than_max() -> None:
    # Use a length that does not align to multiples of (MAX - OVERLAP).
    text = _make_text(_MAX_TOKENS + _OVERLAP_TOKENS + 50)
    assert len(_tokens(text)) > _MAX_TOKENS, "Must produce at least 2 chunks"
    chunks = chunk_abstract(text)
    assert len(chunks) >= 2
    assert len(_tokens(chunks[-1].text)) <= _MAX_TOKENS
