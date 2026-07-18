"""Shared metric computation for the CI baseline/gate — docs/DESIGN.md §6
("PR-smoke workflow fails a deliberately-regressed metric against a stored
baseline") and `ci_baselines` table (docs/DESIGN.md §4).

CI runners here are ephemeral (no long-lived hosted Postgres reachable from
GitHub Actions in this project — decision row 9's "Postgres in CI" assumes a
reachable instance this setup doesn't have), so the baseline that gates PRs
is a checked-in JSON snapshot (results/ci_baseline.json) rather than a live
`ci_baselines` query — same config-next-to-result contract (git_sha +
metric value + producing config), just persisted in git. A real deployment
with a persistent CI database could swap this for `db.get_latest_ci_baseline`
directly; that function already exists and is unit-tested (tests/test_store.py)
for exactly that future swap.

Not every tracked metric actually gates PR-smoke — see
INFORMATIONAL_ONLY_METRICS below for which ones are measured/reported every
PR but don't fail the check, and why a flat point-tolerance can't gate them
meaningfully at n=20/single-run.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from deepresearch.config import current_git_sha
from deepresearch.store import db
from deepresearch.store.models import eval_scores, runs

# (benchmark_name, metric_name in eval_scores) -> flat key in the baseline dict.
# "higher is better" for all of these — cost is handled separately since it's
# "lower is better" and lives on `runs.total_cost_usd`, not `eval_scores`.
_QUALITY_METRICS = {
    ("frames", "accuracy"): "frames.accuracy",
    ("frames", "citation_precision"): "frames.citation_precision",
    ("musique", "answer_f1"): "musique.answer_f1",
    # Raw-report token F1 (above) is crushed by report length regardless of
    # answer quality (docs/RESULTS.md) - this is the same F1 computed against
    # a judge-extracted short answer instead, comparable to MuSiQue's own
    # published (short-answer) baselines. Both are kept: the raw one as the
    # historical/gated number, this one as the literature-comparable one.
    ("musique", "answer_f1_extracted"): "musique.answer_f1_extracted",
}
_COST_BENCHMARKS = ["frames", "musique"]
# docs/DESIGN.md §5.2's own agentic metric ("did the run finish inside
# budget with a synthesized report") — gated here too, not just reported.
# Originally added specifically because CI ran with no LLM key against a
# fake stand-in whose judge verdicts were independent of run content, so
# task_completion_rate was the only axis a real code regression could move
# (see docs/RESULTS.md). CI now runs against a real model (no fake/stub
# fallback exists anymore), so accuracy/citation_precision/answer_f1 are
# real signal too — kept as a gated metric regardless, since "did the run
# finish" is a legitimate agentic metric on its own merits.
_COMPLETION_BENCHMARKS = ["frames", "musique"]

# Session brief's literal tolerances (tighter than CLAUDE.md's placeholder
# 5pt/30% numbers — flagged as a deliberate divergence in docs/RESULTS.md,
# not an oversight; CLAUDE.md's thresholds were themselves marked
# "placeholders until the first real baseline lands").
ACCURACY_DROP_TOLERANCE = 0.03  # 3 points absolute
COST_INCREASE_TOLERANCE = 0.25  # 25% relative

# Quality metrics measured (and shown every PR), but not gated, on PR-smoke.
# PR #7's first real (post-FakeLLMClient-removal) run measured single-run
# noise on this exact metric family up to ~17-25 points at n=20 (repeat-3x
# architecture ablation, docs/RESULTS.md) — a flat point-tolerance can't be
# set below that noise floor without also firing on every PR, and can't be
# set above it without going blind to a real regression of similar size.
# cost_per_query_usd/task_completion_rate stay gated: both are structurally
# low-variance (near-deterministic given a fixed question set and provider)
# and passed clean across every real run so far, fake-client era included.
# The real fix is nightly's own variance-aware policy (CLAUDE.md: "±1 stdev
# of the last 5 nightly runs") gating on a measured distribution instead of
# a single point — not built yet (needs a few more real nightly baselines
# to compute a trustworthy stdev from first), tracked as a follow-up.
INFORMATIONAL_ONLY_METRICS = frozenset(_QUALITY_METRICS.values())


async def compute_current_metrics(database_url: str) -> dict[str, float]:
    """Averages eval_scores + runs.total_cost_usd for whatever benchmark
    rows exist in this database — however many/few questions were just run."""
    engine = db.get_engine(database_url)
    metrics: dict[str, float] = {}

    async with engine.begin() as conn:
        for (benchmark_name, metric_name), key in _QUALITY_METRICS.items():
            result = await conn.execute(
                select(eval_scores.c.value).where(
                    eval_scores.c.benchmark_name == benchmark_name,
                    eval_scores.c.metric_name == metric_name,
                )
            )
            values = [float(row[0]) for row in result.fetchall()]
            if values:
                metrics[key] = sum(values) / len(values)

        for benchmark_name in _COST_BENCHMARKS:
            result = await conn.execute(
                select(runs.c.total_cost_usd).where(runs.c.benchmark_name == benchmark_name)
            )
            values = [float(row[0]) for row in result.fetchall() if row[0] is not None]
            if values:
                metrics[f"{benchmark_name}.cost_per_query_usd"] = sum(values) / len(values)

        for benchmark_name in _COMPLETION_BENCHMARKS:
            result = await conn.execute(select(runs.c.status).where(runs.c.benchmark_name == benchmark_name))
            statuses = [row[0] for row in result.fetchall()]
            if statuses:
                metrics[f"{benchmark_name}.task_completion_rate"] = (
                    sum(1 for status in statuses if status == "completed") / len(statuses)
                )

    return metrics


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, metrics: dict[str, float], *, config: dict | None = None) -> dict:
    baseline = {
        "git_sha": current_git_sha(),
        "metrics": metrics,
        "config": config or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    return baseline


def is_cost_metric(key: str) -> bool:
    return key.endswith(".cost_per_query_usd")


def fmt_metric(key: str, value: float) -> str:
    return f"${value:.4f}" if is_cost_metric(key) else f"{value:.3f}"


def check_regression(key: str, baseline_value: float, current_value: float) -> str | None:
    """Returns a human-readable regression reason, or None if within
    tolerance. Cost metrics regress upward (+25% relative); quality metrics
    regress downward (-3 points absolute) — this session's literal
    tolerances, see the module docstring."""
    if is_cost_metric(key):
        if baseline_value <= 0:
            return None
        increase = (current_value - baseline_value) / baseline_value
        if increase > COST_INCREASE_TOLERANCE:
            return f"cost up {increase * 100:.1f}% (tolerance {COST_INCREASE_TOLERANCE * 100:.0f}%)"
        return None

    drop = baseline_value - current_value
    if drop > ACCURACY_DROP_TOLERANCE:
        return f"dropped {drop * 100:.1f} points (tolerance {ACCURACY_DROP_TOLERANCE * 100:.0f} points)"
    return None


def render_gate_table(baseline_metrics: dict, current_metrics: dict) -> tuple[str, list[str]]:
    """Returns (markdown_table, failure_reasons) — failure_reasons is empty
    iff every *gated* metric present on both sides is within tolerance.
    INFORMATIONAL_ONLY_METRICS are still computed, compared, and shown in
    the table (so drift stays visible every PR) but never contribute to
    failure_reasons — see that constant's comment for why."""
    lines = ["| Metric | Baseline | Current | Delta | Status |", "|---|---|---|---|---|"]
    failures = []
    all_keys = sorted(set(baseline_metrics) | set(current_metrics))

    for key in all_keys:
        baseline_value = baseline_metrics.get(key)
        current_value = current_metrics.get(key)
        if baseline_value is None or current_value is None:
            lines.append(
                f"| `{key}` | {'—' if baseline_value is None else fmt_metric(key, baseline_value)} | "
                f"{'—' if current_value is None else fmt_metric(key, current_value)} | n/a | "
                f"SKIPPED (no data on one side) |"
            )
            continue

        reason = check_regression(key, baseline_value, current_value)
        delta = current_value - baseline_value
        delta_str = f"{delta:+.4f}" if is_cost_metric(key) else f"{delta:+.3f}"
        if not reason:
            status = "OK"
        elif key in INFORMATIONAL_ONLY_METRICS:
            status = f"INFO (not gated on PR-smoke): {reason}"
        else:
            status = f"FAIL: {reason}"
            failures.append(f"`{key}`: {reason}")
        lines.append(
            f"| `{key}` | {fmt_metric(key, baseline_value)} | {fmt_metric(key, current_value)} | "
            f"{delta_str} | {status} |"
        )

    return "\n".join(lines), failures
