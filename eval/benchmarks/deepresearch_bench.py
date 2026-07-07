"""DeepResearch Bench — real RACE scoring + FACT-style citation checking,
gated behind an explicit cost confirmation (docs/DESIGN.md decision row 11).

Adapted from `Ayanami0730/deep_research_bench` (MIT), commit
469cce54ea7f6a63c163d3d9fec879cf289ec484 — pinned the same way FRAMES/MuSiQue
pin their HF dataset revisions (docs/DESIGN.md risk table: benchmark drift).

Two honest departures from the reference implementation, both because this
agent's architecture already provides what the reference's generic pipeline
has to work around:

1. **RACE** uses the reference's real per-task criteria (`criteria.jsonl`,
   fetched and cached below) and its real point-wise scoring prompt/weighted-
   aggregation math (eval/race_judge.py) - faithfully ported. Judge model is
   whichever RunConfig.judge_model is configured for this run, not
   necessarily the reference's GPT-5.5 - see docs/RESULTS.md for what was
   actually used. Scores here are therefore not directly comparable to the
   public DRB leaderboard, which is scored under one fixed evaluator model.

2. **FACT** in the reference implementation extracts claim-URL pairs from
   free-form text and independently re-scrapes each URL (via Jina) to verify
   support, because the systems it benchmarks don't necessarily have
   structured citations. This agent already produces structured
   claim -> source_id citations against already-fetched content
   (eval/metrics/citation.py, already "FACT-protocol style" per CLAUDE.md) -
   reusing that existing, tested checker is methodologically equivalent
   (does the cited source support the claim?) without inventing a second,
   redundant re-scrape pipeline or a new Jina dependency this project has no
   key for. Not a byte-exact FACT port; not comparable to leaderboard FACT
   numbers either.

Runs the real agent against **live Tavily search** (search_backend="tavily"),
not LocalCorpusBackend - DeepResearch Bench tasks are open-ended real-world
research questions with no fixed corpus, unlike FRAMES/MuSiQue.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx  # noqa: E402

from deepresearch.agent.orchestrator import run_research  # noqa: E402
from deepresearch.backends import build_search_backend  # noqa: E402
from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.store import db  # noqa: E402
from deepresearch.telemetry.otel_setup import init_telemetry  # noqa: E402

from eval.judge import Judge  # noqa: E402
from eval.metrics.citation import compute_citation_metrics  # noqa: E402
from eval.race_judge import aggregate_race_score, score_report  # noqa: E402

REPO_COMMIT = "469cce54ea7f6a63c163d3d9fec879cf289ec484"
RAW_BASE = f"https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/{REPO_COMMIT}"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "drb"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# docs/DESIGN.md §5.6 / decision row 11 - researched estimates, not measured
# here (this session measures the real 2-3 question proof-of-mechanics cost
# instead; see docs/RESULTS.md for that real number).
COST_ESTIMATES_USD = {
    "weekly": (15, 35),  # 10-question EN-only subset, reference judges
    "full": (120, 330),  # 100-task suite (50 zh/50 en), monthly/manual only
}


@dataclass
class DRBExample:
    id: int
    topic: str
    prompt: str
    dimension_weight: dict
    criterions: dict


def _fetch_and_cache(filename: str) -> Path:
    path = CACHE_DIR / filename
    if path.exists():
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(f"{RAW_BASE}/data/{'prompt_data' if 'query' in filename else 'criteria_data'}/{filename}", timeout=30)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def load_subset(n: int, seed: int, *, language: str = "en") -> list[DRBExample]:
    """Deterministic sample of real DeepResearch Bench tasks, merged with
    their real official per-task criteria. Fetches+caches both source files
    on first use (data/drb/), reused on every later call."""
    query_path = _fetch_and_cache("query.jsonl")
    criteria_path = _fetch_and_cache("criteria.jsonl")

    queries = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    criteria_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in criteria_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    filtered = [q for q in queries if q["language"] == language]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:n]

    return [
        DRBExample(
            id=q["id"],
            topic=q["topic"],
            prompt=q["prompt"],
            dimension_weight=criteria_by_id[q["id"]]["dimension_weight"],
            criterions=criteria_by_id[q["id"]]["criterions"],
        )
        for q in selected
    ]


def print_cost_estimate(mode: str) -> None:
    low, high = COST_ESTIMATES_USD[mode]
    n_questions = 10 if mode == "weekly" else 100
    print(
        f"\nDeepResearch Bench ({mode}, {n_questions} questions):\n"
        f"  Reference-implementation estimate: ${low}-{high} (RACE: GPT-5.5, FACT: GPT-5.4-mini,\n"
        f"  plus this project's own agent execution cost — see docs/DESIGN.md §5.6).\n"
        f"  This run uses this project's own configured judge_model instead — see\n"
        f"  docs/RESULTS.md for the real measured cost of a small run, likely much lower.\n"
        f"  docs/DESIGN.md decision row 11: this is NOT nightly-affordable regardless.\n"
    )


async def run_and_score_one(example: DRBExample, *, database_url: str) -> dict:
    config = RunConfig(database_url=database_url, search_backend="tavily")
    backend = build_search_backend(config)

    result = await run_research(
        example.prompt, config=config, search_backend=backend, benchmark_name="deepresearch_bench"
    )
    article = result.report.text if result.report else ""

    from deepresearch.llm.client import LLMClient

    llm = LLMClient()
    race_raw, race_usage = await score_report(
        llm,
        model=config.judge_model,
        task_prompt=example.prompt,
        article=article,
        criteria_data={"dimension_weight": example.dimension_weight, "criterions": example.criterions},
    )
    race_scores = aggregate_race_score(race_raw, {"dimension_weight": example.dimension_weight, "criterions": example.criterions})

    judge = Judge(llm, config)
    citation_metrics = await compute_citation_metrics(result, judge)

    return {
        "task_id": example.id,
        "run_id": result.run_id,
        "status": result.status.value,
        "agent_cost_usd": result.total_cost_usd,
        "race_judge_cost_usd": race_usage.cost_usd,
        "citation_judge_cost_usd": judge.total_cost_usd,
        "race_scores": race_scores,
        "citation_coverage": citation_metrics.coverage,
        "citation_precision": citation_metrics.precision,
    }


async def run_subset(n: int, seed: int, *, database_url: str) -> dict:
    init_telemetry()
    await db.ensure_schema(database_url)
    examples = load_subset(n, seed)
    print(f"[deepresearch_bench] {len(examples)} real EN tasks, live Tavily search + real judge scoring.")

    results = []
    start = time.monotonic()

    for ex in examples:
        print(f"  running task {ex.id}: {ex.prompt[:80]}...")
        r = await run_and_score_one(ex, database_url=database_url)
        results.append(r)
        task_scores = [
            {
                "run_id": r["run_id"],
                "benchmark_name": "deepresearch_bench",
                "question_id": str(r["task_id"]),
                "metric_name": f"race_{dim}",
                "value": value,
                "judge_model": None,
                "rubric_version": None,
                "raw_judge_output": None,
            }
            for dim, value in r["race_scores"].items()
        ] + [
            {
                "run_id": r["run_id"], "benchmark_name": "deepresearch_bench", "question_id": str(r["task_id"]),
                "metric_name": "citation_coverage", "value": r["citation_coverage"],
                "judge_model": None, "rubric_version": None, "raw_judge_output": None,
            },
            {
                "run_id": r["run_id"], "benchmark_name": "deepresearch_bench", "question_id": str(r["task_id"]),
                "metric_name": "citation_precision", "value": r["citation_precision"],
                "judge_model": None, "rubric_version": None, "raw_judge_output": None,
            },
        ]
        # Flushed per-task, not batched to the end of the loop — same
        # failure mode already documented (and avoided) in run_musique/
        # run_frames: a crash on task k (e.g. a sustained OpenRouter 429
        # exhausting the retry budget — confirmed live, this exact run)
        # must not discard the already-computed, already-paid-for scores
        # for tasks before it. This bug did exactly that on its first real
        # run — task 76's RACE+citation scores were computed, then lost
        # when task 74's judge call crashed before the end-of-loop flush.
        await db.bulk_insert_eval_scores(database_url, task_scores)

    elapsed = time.monotonic() - start

    total_agent_cost = sum(r["agent_cost_usd"] for r in results)
    total_judge_cost = sum(r["race_judge_cost_usd"] + r["citation_judge_cost_usd"] for r in results)
    mean_race_total = sum(r["race_scores"].get("total", 0.0) for r in results) / len(results) if results else 0.0

    summary = {
        "n": len(examples),
        "seed": seed,
        "wall_clock_seconds": elapsed,
        "mean_race_total": mean_race_total,
        "mean_citation_coverage": sum(r["citation_coverage"] for r in results) / len(results) if results else 0.0,
        "mean_citation_precision": sum(r["citation_precision"] for r in results) / len(results) if results else 0.0,
        "total_agent_cost_usd": total_agent_cost,
        "total_judge_cost_usd": total_judge_cost,
        "total_cost_usd": total_agent_cost + total_judge_cost,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"deepresearch_bench_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["weekly", "full"], default="weekly")
    parser.add_argument("--n", type=int, default=None, help="override the weekly(10)/full(100) question count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-url", type=str, default=None)
    parser.add_argument("--confirm", action="store_true", help="required to proceed past the cost estimate")
    args = parser.parse_args()

    print_cost_estimate(args.mode)

    if not args.confirm:
        print("Not running — pass --confirm to proceed past this cost estimate.")
        sys.exit(1)

    n = args.n if args.n is not None else (10 if args.mode == "weekly" else 100)
    database_url = args.database_url or RunConfig().database_url
    asyncio.run(run_subset(n, args.seed, database_url=database_url))


if __name__ == "__main__":
    main()
