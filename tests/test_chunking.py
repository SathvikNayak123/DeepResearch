from __future__ import annotations

from deepresearch.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", chunk_size=800) == ["hello world"]


def test_long_text_produces_multiple_bounded_chunks():
    text = "".join(str(i) for i in range(1000))  # long, position-distinguishable text
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)
    assert text.startswith(chunks[0][:700])  # first chunk anchored at text start


def test_overlap_shares_content_between_consecutive_chunks():
    text = "".join(str(i) for i in range(1000))
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    first_tail = chunks[0][-100:]
    second_head = chunks[1][:100]
    assert first_tail == second_head
