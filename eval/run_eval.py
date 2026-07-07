"""Main eval harness entrypoint — docs/DESIGN.md session 4.

Runs a benchmark subset end-to-end through the real agent
(deepresearch.agent.orchestrator.run_research) against LocalCorpusBackend,
scores each question, and writes eval_scores rows to config.database_url —
the run store this session wired up. Every run (including this one) writes
a `runs` row automatically; nothing extra to do here for that part.

Usage:
    python -m eval.run_eval --benchmark musique --n 20 --seed 42
    python -m eval.run_eval --benchmark frames --n 20 --seed 42
    python -m eval.run_eval --mode smoke   # both benchmarks, n=20 each
    python -m eval.run_eval --mode full    # both benchmarks, n=100 each
    python -m eval.run_eval --reliability --n 20 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepresearch.agent.orchestrator import run_research  # noqa: E402
from deepresearch.backends.local_corpus import LocalCorpusBackend  # noqa: E402
from deepresearch.config import RunConfig, current_git_sha  # noqa: E402
from deepresearch.llm.client import LLMClient  # noqa: E402
from deepresearch.store import db  # noqa: E402
from deepresearch.telemetry.otel_setup import init_telemetry  # noqa: E402

from eval.benchmarks import frames as frames_bench  # noqa: E402
from eval.benchmarks import musique as musique_bench  # noqa: E402
from eval.fake_llm import FakeLLMClient  # noqa: E402
from eval.judge import Judge, estimate_judge_cost_usd  # noqa: E402
from eval.metrics.answer_f1 import best_answer_f1, gold_contained  # noqa: E402
from eval.metrics.citation import compute_citation_metrics  # noqa: E402
from eval.metrics.reliability import compute_reliability  # noqa: E402
from eval.metrics.trajectory import compute_trajectory_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Rough planning estimate for the agent's own LLM cost per question (plan +
# N workers + reflection + synthesis, claude-opus-4-8 pricing) — refine once
# real runs give a measured average to replace this guess with. Judge cost
# is estimated separately (eval.judge.estimate_judge_cost_usd), since it's a
# different, cheaper model with its own per-call token profile.
ESTIMATED_AGENT_COST_PER_QUESTION_USD = 0.15


def make_llm() -> tuple[object, bool]:
    """Returns (llm_client, is_real_llm). Falls back to FakeLLMClient with a
    loud banner if ANTHROPIC_API_KEY isn't set, so the harness stays
    runnable without live credentials — see docs/RESULTS.md for what that
    means for any numbers produced this way."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMClient(), True
    print(
        "\n*** No ANTHROPIC_API_KEY set - using FakeLLMClient. ***\n"
        "*** Scores below are HARNESS VALIDATION, not a real model baseline. ***\n"
        "*** See docs/RESULTS.md for what's real vs. simulated in this run. ***\n"
    )
    return FakeLLMClient(seed=0), False


async def _run_one_question(
    question: str,
    corpus_path,
    *,
    database_url: str,
    llm,
    benchmark_name: str,
):
    config = RunConfig(
        database_url=database_url,
        search_backend="local_corpus",
        local_corpus_dir=str(corpus_path),
        cache_enabled=False,  # eval runs are cold by design — this session's bypass flag
    )
    backend = LocalCorpusBackend.from_json_file(corpus_path)
    return await run_research(question, config=config, search_backend=backend, llm=llm, benchmark_name=benchmark_name)


async def run_musique(n: int, seed: int, *, database_url: str) -> dict:
    examples = musique_bench.load_subset(n, seed)
    llm, is_real = make_llm()

    print(f"[musique] {len(examples)} questions. Estimated agent cost: "
          f"${len(examples) * ESTIMATED_AGENT_COST_PER_QUESTION_USD:.2f} "
          f"(no judge calls — MuSiQue is scored by Answer F1, string-based)")

    results = []
    scores_rows = []
    tool_calls_by_run: dict[str, list[dict]] = {}
    start = time.monotonic()

    for ex in examples:
        result = await _run_one_question(
            ex.question, ex.corpus_path, database_url=database_url, llm=llm, benchmark_name="musique"
        )
        results.append(result)

        predicted = result.report.text if result.report else ""
        f1 = best_answer_f1(predicted, ex.gold_answers)
        contains_gold = float(any(gold_contained(predicted, g) for g in ex.gold_answers))

        question_scores = [
            _score_row(result.run_id, "musique", ex.question_id, "answer_f1", f1),
            _score_row(result.run_id, "musique", ex.question_id, "answer_contains_gold", contains_gold),
        ]
        # Flushed per-question, not batched to the end of the loop: a crash on
        # question k (a downstream API error, a budget blowout) must not discard
        # the k-1 questions' already-computed scores — same failure mode already
        # documented for RunRecorder's end-of-run trajectory flush, one layer up.
        await db.bulk_insert_eval_scores(database_url, question_scores)
        scores_rows.extend(question_scores)

        tool_calls_by_run[result.run_id] = await db.get_tool_calls_for_run(database_url, result.run_id)

    elapsed = time.monotonic() - start
    traj = compute_trajectory_metrics(results, tool_calls_by_run)

    return {
        "benchmark": "musique",
        "n": len(examples),
        "is_real_llm": is_real,
        "wall_clock_seconds": elapsed,
        "mean_answer_f1": _mean(scores_rows, "answer_f1"),
        "mean_answer_contains_gold": _mean(scores_rows, "answer_contains_gold"),
        "trajectory": traj.summary(),
        "total_agent_cost_usd": sum(r.total_cost_usd for r in results),
    }


async def run_frames(n: int, seed: int, *, database_url: str) -> dict:
    rows = frames_bench.load_subset(n, seed)
    to_fetch = frames_bench.estimate_articles_to_fetch(rows)
    print(f"[frames] {len(rows)} questions selected. Will fetch ~{to_fetch} new Wikipedia "
          f"articles (already-cached ones are skipped) before running.")
    examples = await frames_bench.ingest_corpus(rows)

    llm, is_real = make_llm()
    judge_config = RunConfig(database_url=database_url)
    judge = Judge(llm, judge_config)

    est_judge_calls = len(examples) * 2  # accuracy + ~1 citation check on average, rough
    print(
        f"[frames] Estimated cost: agent ~${len(examples) * ESTIMATED_AGENT_COST_PER_QUESTION_USD:.2f} "
        f"+ judge ~${estimate_judge_cost_usd(est_judge_calls, judge_config.judge_model):.4f} "
        f"({judge_config.judge_model})"
    )

    results = []
    scores_rows = []
    tool_calls_by_run: dict[str, list[dict]] = {}
    start = time.monotonic()

    for ex in examples:
        result = await _run_one_question(
            ex.question, ex.corpus_path, database_url=database_url, llm=llm, benchmark_name="frames"
        )
        results.append(result)

        predicted = result.report.text if result.report else ""
        correct, rationale, _ = await judge.judge_accuracy(
            question=ex.question, gold_answer=ex.answer, predicted_answer=predicted
        )
        citation = await compute_citation_metrics(result, judge)

        question_scores = [
            _score_row(
                result.run_id, "frames", ex.example_id, "accuracy", float(correct),
                judge_model=judge_config.judge_model, rubric_version=judge_config.judge_rubric_version,
                raw_judge_output={"rationale": rationale},
            ),
            _score_row(result.run_id, "frames", ex.example_id, "citation_coverage", citation.coverage),
            _score_row(
                result.run_id, "frames", ex.example_id, "citation_precision", citation.precision,
                judge_model=judge_config.judge_model, rubric_version=judge_config.judge_rubric_version,
            ),
        ]
        # Flushed per-question — see run_musique's identical comment: a crash
        # partway through the loop (budget blowout, an API error) must not
        # discard scores already computed for the questions before it.
        await db.bulk_insert_eval_scores(database_url, question_scores)
        scores_rows.extend(question_scores)

        tool_calls_by_run[result.run_id] = await db.get_tool_calls_for_run(database_url, result.run_id)

    elapsed = time.monotonic() - start
    traj = compute_trajectory_metrics(results, tool_calls_by_run)

    return {
        "benchmark": "frames",
        "n": len(examples),
        "is_real_llm": is_real,
        "wall_clock_seconds": elapsed,
        "mean_accuracy": _mean(scores_rows, "accuracy"),
        "mean_citation_coverage": _mean(scores_rows, "citation_coverage"),
        "mean_citation_precision": _mean(scores_rows, "citation_precision"),
        "trajectory": traj.summary(),
        "total_agent_cost_usd": sum(r.total_cost_usd for r in results),
        "judge_calls_made": judge.calls_made,
        "judge_cache_hits": judge.cache_hits,
        "judge_actual_cost_usd": judge.total_cost_usd,
    }


async def run_reliability(n: int, repeats: int, seed: int, *, database_url: str) -> dict:
    """docs/DESIGN.md §5.2 + CLAUDE.md: repeat a fixed subset 3-5x, report
    the distribution (variance) and an all-consistent (pass^k) rate — never
    a single point estimate."""
    examples = musique_bench.load_subset(n, seed)
    llm, is_real = make_llm()

    print(f"[reliability] {len(examples)} questions x {repeats} repeats = "
          f"{len(examples) * repeats} runs. Estimated agent cost: "
          f"${len(examples) * repeats * ESTIMATED_AGENT_COST_PER_QUESTION_USD:.2f}")

    per_question_correct: dict[str, list[bool]] = {ex.question_id: [] for ex in examples}
    total_cost = 0.0

    for _repeat in range(repeats):
        for ex in examples:
            result = await _run_one_question(
                ex.question, ex.corpus_path, database_url=database_url, llm=llm, benchmark_name="musique_reliability"
            )
            total_cost += result.total_cost_usd
            predicted = result.report.text if result.report else ""
            correct = any(gold_contained(predicted, g) for g in ex.gold_answers)
            per_question_correct[ex.question_id].append(correct)

    report = compute_reliability(per_question_correct)
    summary = {"is_real_llm": is_real, "total_agent_cost_usd": total_cost, **report.summary()}

    await db.bulk_insert_eval_scores(
        database_url,
        [
            _score_row(f"reliability-{current_git_sha()}", "reliability", None, key, value)
            for key, value in (
                ("mean_accuracy", report.mean_accuracy),
                ("stdev_accuracy", report.stdev_accuracy),
                ("all_consistent_rate", report.all_consistent_rate),
            )
        ],
    )
    return summary


def _score_row(run_id, benchmark_name, question_id, metric_name, value, *, judge_model=None, rubric_version=None, raw_judge_output=None):
    return {
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "question_id": question_id,
        "metric_name": metric_name,
        "value": value,
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "raw_judge_output": raw_judge_output,
    }


def _mean(rows: list[dict], metric_name: str) -> float:
    values = [r["value"] for r in rows if r["metric_name"] == metric_name]
    return sum(values) / len(values) if values else 0.0


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=["frames", "musique", "both"], default="both")
    parser.add_argument("--mode", choices=["smoke", "full"], default=None, help="20q (smoke) or 100q (full) both benchmarks")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reliability", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--database-url", type=str, default=None)
    args = parser.parse_args()

    init_telemetry()
    database_url = args.database_url or RunConfig().database_url
    await db.ensure_schema(database_url)

    if args.reliability:
        summary = await run_reliability(args.n, args.repeats, args.seed, database_url=database_url)
        _write_and_print("reliability", summary)
        return

    n = {"smoke": 20, "full": 100}.get(args.mode, args.n)
    benchmarks = ["frames", "musique"] if args.benchmark == "both" else [args.benchmark]

    summaries = {}
    for name in benchmarks:
        if name == "frames":
            summaries["frames"] = await run_frames(n, args.seed, database_url=database_url)
        else:
            summaries["musique"] = await run_musique(n, args.seed, database_url=database_url)

    _write_and_print(args.mode or "custom", summaries)


def _write_and_print(label: str, summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{label}_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
