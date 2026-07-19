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
import uuid
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


_PROVIDER_REQUIRED_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # bedrock has no single required env var — it authenticates via the
    # ambient AWS credential chain (boto3), not a stored API key.
}


def make_llm() -> tuple[object, bool]:
    """Returns (llm_client, is_real_llm) — always a real LLMClient. No
    fake/stub fallback: docs/DESIGN.md's own premise ("every architectural
    box has a metric attached") means an eval score must come from a real
    model or not be produced at all. Fails loudly and immediately if the
    configured provider's key is missing, instead of surfacing whatever
    error the SDK happens to raise several calls deep."""
    provider = os.environ.get("DEEPRESEARCH_LLM_PROVIDER", "anthropic")
    required_key = _PROVIDER_REQUIRED_KEY.get(provider)
    if required_key and not os.environ.get(required_key):
        raise RuntimeError(
            f"DEEPRESEARCH_LLM_PROVIDER={provider!r} requires {required_key} to be set. "
            "eval.run_eval only runs against a real model — no fake/stub fallback. "
            "Set the key (or point DEEPRESEARCH_LLM_PROVIDER at a provider you have "
            "credentials for) before running."
        )
    return LLMClient(), True


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


async def run_musique(n: int, seed: int, *, database_url: str, max_total_cost_usd: float | None = None) -> dict:
    examples = musique_bench.load_subset(n, seed)
    llm, is_real = make_llm()

    # Extraction is a cheap judge-model call per question (docs/RESULTS.md:
    # raw-report answer_f1 is crushed by report length, not answer quality —
    # a real example scored 0.05 despite stating the gold fact correctly).
    judge_config = RunConfig(database_url=database_url)
    judge = Judge(llm, judge_config)

    # Resume support: skip questions this exact database_url already has a
    # real score for, so re-invoking with the same --n/--seed after an
    # interruption doesn't re-run (and re-charge real LLM cost for)
    # already-completed questions -- load_subset() is deterministic, so a
    # naive re-run would reprocess the same prefix every time.
    already_scored = await db.get_scored_question_ids(database_url, "musique")
    todo = [ex for ex in examples if ex.question_id not in already_scored]
    if len(todo) < len(examples):
        print(f"[musique] resuming: {len(examples) - len(todo)} of {len(examples)} already scored, skipping those")

    print(f"[musique] {len(todo)} questions to run. Estimated agent cost: "
          f"${len(todo) * ESTIMATED_AGENT_COST_PER_QUESTION_USD:.2f} "
          f"+ extraction ~${estimate_judge_cost_usd(len(todo), judge_config.judge_model):.4f} "
          f"({judge_config.judge_model}) — MuSiQue is scored by Answer F1 (string-based) "
          f"both against the raw report and against an extracted short answer")

    results = []
    scores_rows = []
    tool_calls_by_run: dict[str, list[dict]] = {}
    failed_question_ids: list[str] = []
    start = time.monotonic()

    for ex in todo:
        if max_total_cost_usd is not None:
            spent = await db.get_total_cost_usd(database_url)
            if spent >= max_total_cost_usd:
                print(f"[musique] stopping: cumulative spend ${spent:.4f} >= cap ${max_total_cost_usd:.2f} "
                      f"({len(examples) - len(results) - len(failed_question_ids)} questions not attempted)")
                break

        # One question's exception must not discard every question after it
        # in the batch (confirmed live: a real FRAMES question's planner
        # over-generated a plan, PlanValidationError propagated uncaught all
        # the way to the top of run_research(), and killed an n=30 run after
        # only 19 questions -- with n=100+ runs costing hours, losing the
        # whole remaining batch over one question is exactly the failure
        # mode this must not have). Already-flushed scores for prior
        # questions are unaffected either way (see the per-question flush
        # below); this addresses the batch *continuing*, not data survival.
        try:
            result = await _run_one_question(
                ex.question, ex.corpus_path, database_url=database_url, llm=llm, benchmark_name="musique"
            )
        except Exception as exc:
            print(f"[musique] question {ex.question_id} failed, skipping: {exc!r}")
            failed_question_ids.append(ex.question_id)
            continue
        results.append(result)

        predicted = result.report.text if result.report else ""
        f1 = best_answer_f1(predicted, ex.gold_answers)
        contains_gold = float(any(gold_contained(predicted, g) for g in ex.gold_answers))

        short_answer, _ = await judge.extract_short_answer(question=ex.question, report_text=predicted)
        f1_extracted = best_answer_f1(short_answer, ex.gold_answers)

        question_scores = [
            _score_row(result.run_id, "musique", ex.question_id, "answer_f1", f1),
            _score_row(result.run_id, "musique", ex.question_id, "answer_contains_gold", contains_gold),
            _score_row(result.run_id, "musique", ex.question_id, "answer_f1_extracted", f1_extracted),
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
        "n_target": len(examples),
        "n": len(todo),  # what THIS invocation ran -- matches wall_clock/cost below
        "n_failed": len(failed_question_ids),
        "failed_question_ids": failed_question_ids,
        "is_real_llm": is_real,
        "wall_clock_seconds": elapsed,
        "mean_answer_f1": _mean(scores_rows, "answer_f1"),
        "mean_answer_contains_gold": _mean(scores_rows, "answer_contains_gold"),
        "mean_answer_f1_extracted": _mean(scores_rows, "answer_f1_extracted"),
        "trajectory": traj.summary(),
        "total_agent_cost_usd": sum(r.total_cost_usd for r in results),
        "extraction_calls_made": judge.calls_made,
        "extraction_cache_hits": judge.cache_hits,
        "extraction_actual_cost_usd": judge.total_cost_usd,
    }


async def run_frames(n: int, seed: int, *, database_url: str, max_total_cost_usd: float | None = None) -> dict:
    rows = frames_bench.load_subset(n, seed)
    to_fetch = frames_bench.estimate_articles_to_fetch(rows)
    print(f"[frames] {len(rows)} questions selected. Will fetch ~{to_fetch} new Wikipedia "
          f"articles (already-cached ones are skipped) before running.")
    examples = await frames_bench.ingest_corpus(rows)

    llm, is_real = make_llm()
    judge_config = RunConfig(database_url=database_url)
    judge = Judge(llm, judge_config)

    # Resume support — see run_musique's identical comment.
    already_scored = await db.get_scored_question_ids(database_url, "frames")
    todo = [ex for ex in examples if ex.example_id not in already_scored]
    if len(todo) < len(examples):
        print(f"[frames] resuming: {len(examples) - len(todo)} of {len(examples)} already scored, skipping those")

    est_judge_calls = len(todo) * 2  # accuracy + ~1 citation check on average, rough
    print(
        f"[frames] {len(todo)} questions to run. Estimated cost: agent ~${len(todo) * ESTIMATED_AGENT_COST_PER_QUESTION_USD:.2f} "
        f"+ judge ~${estimate_judge_cost_usd(est_judge_calls, judge_config.judge_model):.4f} "
        f"({judge_config.judge_model})"
    )

    results = []
    scores_rows = []
    tool_calls_by_run: dict[str, list[dict]] = {}
    failed_question_ids: list[str] = []
    start = time.monotonic()

    for ex in todo:
        if max_total_cost_usd is not None:
            spent = await db.get_total_cost_usd(database_url)
            if spent >= max_total_cost_usd:
                print(f"[frames] stopping: cumulative spend ${spent:.4f} >= cap ${max_total_cost_usd:.2f} "
                      f"({len(examples) - len(results) - len(failed_question_ids)} questions not attempted)")
                break

        # See run_musique's identical try/except for why: one question's
        # exception must not discard every question after it in the batch.
        try:
            result = await _run_one_question(
                ex.question, ex.corpus_path, database_url=database_url, llm=llm, benchmark_name="frames"
            )
        except Exception as exc:
            print(f"[frames] question {ex.example_id} failed, skipping: {exc!r}")
            failed_question_ids.append(ex.example_id)
            continue
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
        "n_target": len(examples),
        "n": len(todo),  # what THIS invocation ran -- matches wall_clock/cost below
        "n_failed": len(failed_question_ids),
        "failed_question_ids": failed_question_ids,
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

    # These 3 summary metrics aren't tied to any single question's run, but
    # eval_scores.run_id is a real Postgres UUID column with a FK to runs.run_id
    # (SQLite's loose typing/FK enforcement let f"reliability-{git_sha}" - not
    # UUID-shaped, not a real runs row - silently pass here before; Postgres
    # rejects it outright). Anchor to a dedicated synthetic runs row instead.
    reliability_run_id = uuid.uuid4().hex
    await db.create_run(
        database_url,
        run_id=reliability_run_id,
        benchmark_name="reliability",
        config={"n": n, "repeats": repeats, "seed": seed},
        git_sha=current_git_sha(),
        status="completed",
    )
    await db.bulk_insert_eval_scores(
        database_url,
        [
            _score_row(reliability_run_id, "reliability", None, key, value)
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
    parser.add_argument(
        "--max-total-cost-usd", type=float, default=None,
        help="Stop starting new questions once cumulative real spend in this database_url (across benchmarks/"
             "invocations) reaches this. Checked against actual measured cost per question, not the upfront "
             "estimate -- a hard safety cap, not a projection.",
    )
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
            summaries["frames"] = await run_frames(
                n, args.seed, database_url=database_url, max_total_cost_usd=args.max_total_cost_usd
            )
        else:
            summaries["musique"] = await run_musique(
                n, args.seed, database_url=database_url, max_total_cost_usd=args.max_total_cost_usd
            )

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
