from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_eval_prompt(name: str) -> str:
    """Versioned judge prompts live in files, not inline strings — same
    convention as src/deepresearch/prompts (see that package's loader)."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")
