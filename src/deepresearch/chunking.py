from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split fetched page text into overlapping character-window chunks.

    Character windows, not token-aware — the reranker truncates to its own
    max sequence length regardless, and this keeps chunking backend-agnostic.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def cap_chunks(chunks: list[str], max_chunks: int) -> list[str]:
    """Evenly-spaced subsample down to at most max_chunks, preserving order.

    Full documents (e.g. FRAMES' complete Wikipedia articles, tens of
    thousands of characters) chunk into dozens of ~800-char windows; scoring
    every one of them is where the self-hosted CPU cross-encoder's cost
    balloons (docs/RESULTS.md: FRAMES averaged 146.5 rerank candidates/call
    vs. MuSiQue's ~7, and a live run measured ~130-400s per rerank call at
    that scale). Sampling evenly across the whole document — not just
    truncating to the head — keeps some coverage of later sections instead
    of only ever seeing the lead paragraph.
    """
    n = len(chunks)
    if max_chunks <= 0 or n <= max_chunks:
        return chunks
    if max_chunks == 1:
        return [chunks[0]]
    step = (n - 1) / (max_chunks - 1)
    indices = sorted({round(i * step) for i in range(max_chunks)})
    return [chunks[i] for i in indices]
