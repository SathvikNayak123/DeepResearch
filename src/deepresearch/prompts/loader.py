from __future__ import annotations

from importlib import resources


def load_prompt(name: str) -> str:
    """Load a versioned prompt file (e.g. "planner_v1.txt") — prompts live
    in files, not inline strings, so new versions ship as new files."""
    return resources.files("deepresearch.prompts").joinpath(name).read_text(encoding="utf-8")
