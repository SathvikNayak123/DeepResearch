from __future__ import annotations

from deepresearch.chunking import cap_chunks, chunk_text


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


def test_cap_chunks_is_noop_under_the_limit():
    chunks = ["a", "b", "c"]
    assert cap_chunks(chunks, 10) == chunks


def test_cap_chunks_bounds_count_and_preserves_order():
    chunks = [str(i) for i in range(30)]
    capped = cap_chunks(chunks, 10)

    assert len(capped) <= 10
    assert capped == sorted(capped, key=int)  # ascending -> order preserved


def test_cap_chunks_includes_first_and_last_for_coverage():
    chunks = [str(i) for i in range(30)]
    capped = cap_chunks(chunks, 10)

    assert capped[0] == "0"
    assert capped[-1] == "29"


def test_cap_chunks_single_slot_keeps_first_chunk():
    chunks = ["a", "b", "c"]
    assert cap_chunks(chunks, 1) == ["a"]
