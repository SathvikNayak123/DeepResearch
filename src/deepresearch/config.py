from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Every entrypoint (cli.py, api/main.py, eval/run_eval.py) imports RunConfig
# from here, so loading .env once at import time covers all of them. Doesn't
# override already-exported env vars (python-dotenv default), so CI/shell
# exports still win over a stale .env.
load_dotenv()


@lru_cache(maxsize=1)
def current_git_sha() -> str:
    """Best-effort git SHA for config-next-to-result stamping. Falls back to
    "no-git" rather than raising — this repo may not be a git checkout in
    every environment it runs in (e.g. a fresh sandbox)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except Exception:
        return "no-git"


@dataclass
class BudgetConfig:
    """Explicit, configurable stopping criteria. "The agent decides" is not one."""

    max_replans: int = 2
    max_total_tokens: int = 200_000
    max_wall_clock_seconds: float = 600.0
    max_usd: float = 5.0


@dataclass
class RunConfig:
    max_workers: int = 4
    coverage_threshold: float = 0.8
    planner_model: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_PLANNER_MODEL", "claude-opus-4-8"))
    worker_model: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_WORKER_MODEL", "claude-opus-4-8"))
    reflection_model: str = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_REFLECTION_MODEL", "claude-opus-4-8")
    )
    synthesis_model: str = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_SYNTHESIS_MODEL", "claude-opus-4-8")
    )
    search_backend: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_SEARCH_BACKEND", "tavily"))
    local_corpus_dir: str | None = field(default_factory=lambda: os.getenv("DEEPRESEARCH_LOCAL_CORPUS_DIR"))
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Planning style (docs/DESIGN.md decision row 2). "plan_first" is the
    # default per that decision; "react" is the alternative this session's
    # ablation measures head-to-head — see docs/DESIGN.md's dated addendum.
    planning_style: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_PLANNING_STYLE", "plan_first"))
    max_react_steps: int = 4  # bounded step ceiling for "react" mode, same role as max_replans

    # Retrieval quality (docs/DESIGN.md decision row 7). Default set by the
    # rerank ablation in docs/RESULTS.md, not a guess — see that doc before
    # changing it.
    rerank_enabled: bool = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_RERANK_ENABLED", "true").lower() != "false"
    )
    rerank_backend: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_RERANK_BACKEND", "bge"))
    candidate_pool_size: int = 6  # search results fetched before reranking
    rerank_top_k: int = 6  # chunks kept for the worker's LLM context after reranking
    # Caps chunks-per-source fed into the reranker (docs/RESULTS.md: a full
    # Wikipedia article can chunk into dozens of ~800-char windows; scoring
    # every one is why FRAMES rerank calls measured 130-400s+ each in a real
    # run). A no-op for already-short, pre-chunked documents (MuSiQue).
    max_chunks_per_source: int = field(
        default_factory=lambda: int(os.getenv("DEEPRESEARCH_MAX_CHUNKS_PER_SOURCE", "10"))
    )

    # Caching (docs/DESIGN.md decision row 8). A cache, not agent memory —
    # no cross-run reasoning state, only raw search/fetch payload reuse.
    # cache_enabled is the one-flag bypass eval runs use to force cold.
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_CACHE_ENABLED", "true").lower() != "false"
    )
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    search_cache_ttl_seconds: int = 24 * 3600  # 24h — search results
    fetch_cache_ttl_seconds: int = 7 * 24 * 3600  # 7d — extracted pages
    # Tavily pay-as-you-go: $0.008/credit; basic search = 1 credit,
    # extract = 1 credit per 5 URLs (=> ~1/5 credit per single fetch call).
    search_cost_usd: float = 0.008
    fetch_cost_usd: float = 0.008 / 5

    # Run store (docs/DESIGN.md decision row 9). SQLite by default — the
    # documented dev-loop swap-in; point at postgresql+asyncpg://... in
    # CI/deployed (docker-compose's `postgres` service, or DATABASE_URL env).
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./deepresearch.db")
    )

    # Judge economics (docs/DESIGN.md §5.5). Cheap tier by default, distinct
    # from the agent's own model — this is a deliberate exception to
    # "always use the strongest model," made explicitly for cost control on
    # a task (grading) that doesn't need frontier capability.
    judge_model: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_JUDGE_MODEL", "claude-haiku-4-5"))
    judge_rubric_version: str = "v1"

    @classmethod
    def from_overrides(cls, overrides: dict | None = None) -> "RunConfig":
        cfg = cls()
        if not overrides:
            return cfg
        overrides = dict(overrides)
        budget_overrides = overrides.pop("budget", None)
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if budget_overrides:
            for key, value in budget_overrides.items():
                if hasattr(cfg.budget, key):
                    setattr(cfg.budget, key, value)
        return cfg
