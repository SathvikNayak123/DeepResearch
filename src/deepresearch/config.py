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
    # Bounded parallel wave width (docs/DESIGN.md decision row 1): the
    # supervisor's readiness-gated Send fan-out never dispatches more than
    # this many subagents in one wave (config.max_concurrency).
    max_workers: int = 4
    # Planner over-decomposition guard (docs/DESIGN.md decision row 1/2): a
    # dependency-graph plan validated against this ceiling before dispatch —
    # see agent/dag.py's validate_plan.
    max_nodes: int = 6
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

    # Per-node ReAct step ceiling (agent/subagent.py's tool-calling loop) —
    # same explicit-gate role as max_replans/max_corrections below.
    max_react_steps: int = 5
    # Bounded per-hop self-correction (requeue-on-failed-verify) passes
    # (agent/graph.py's subagent_node) — an explicit stopping gate, never
    # "the agent decides".
    max_corrections: int = 2
    # Bounded post-synthesis reflection passes (agent/graph.py): the final
    # report can trigger at most this many follow-up-node waves before the
    # run stops regardless of remaining gaps — an explicit ceiling, same
    # role as max_replans/max_corrections above.
    max_reflect: int = 2

    # LangGraph checkpointer (docs/DESIGN.md decision row 14): an
    # AsyncSqliteSaver keyed by thread_id=run_id, so an interrupted run can
    # resume from its last completed wave instead of restarting from
    # scratch. Distinct from database_url below (the eval/run-store DB) —
    # this is LangGraph's own state-snapshot store.
    checkpoint_db_path: str = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_CHECKPOINT_DB", "./checkpoints.sqlite")
    )

    # Retrieval quality (docs/DESIGN.md decision row 7). Default set by the
    # rerank ablation in docs/RESULTS.md, not a guess — see that doc before
    # changing it.
    rerank_enabled: bool = field(
        default_factory=lambda: os.getenv("DEEPRESEARCH_RERANK_ENABLED", "true").lower() != "false"
    )
    rerank_backend: str = field(default_factory=lambda: os.getenv("DEEPRESEARCH_RERANK_BACKEND", "bge"))
    # Search results (documents) fetched before reranking. Was 6 — too narrow
    # to even structurally cover FRAMES questions, which docs/DESIGN.md §5.1
    # documents as needing 2-15 Wikipedia articles: at pool_size=6, a question
    # needing 7+ relevant articles can never retrieve them all regardless of
    # rerank quality, since candidates not fetched can't be reranked into
    # existence. Widened to give the cross-encoder real room to correct the
    # search backend's own (cheaper, first-stage) ranking mistakes, not just
    # reorder an already-narrow set. Tradeoff, not free: more fetch calls
    # (cost/latency) and more chunks entering the same self-hosted CPU
    # reranker call that docs/DESIGN.md row 7 already flags as the dominant
    # p95 latency contributor — re-measure rerank latency at this pool size
    # before shipping it live (docs/RESULTS.md's CPU-oversubscription finding
    # was hit at a smaller chunk volume than this).
    candidate_pool_size: int = 20
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
