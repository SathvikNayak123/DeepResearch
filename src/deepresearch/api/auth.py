from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Single shared demo key, not multi-tenant auth (docs/DESIGN.md non-goals).
    No-op if DEEPRESEARCH_API_KEY is unset, so local/dev/CI need no header.

    Lives here (not in main.py) so main.py, streaming.py, and routes_runs.py
    can all depend on it without a circular import between them.
    """
    expected = os.getenv("DEEPRESEARCH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
