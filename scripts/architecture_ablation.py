"""Architecture ablation — docs/DESIGN.md decision rows 1 (topology) and 2
(planning style), "the ablation that powers your 'I chose X over Y' story."

Same MuSiQue smoke subset, three variants:
  1. plan_first_pool4 — the current default (bounded parallel worker pool,
     max_workers=4, upfront plan).
  2. plan_first_pool1 — worker-pool-size sweep: same plan-first planning,
     pool collapsed to a single sequential worker (row 1's "does parallelism
     actually buy anything" question).
  3. react — interleaved ReAct (row 2's alternative): no upfront plan, one
     query decided at a time from the claims gathered so far, sequential.

Findings (including anything that contradicts the original choice) get
written into docs/DESIGN.md as a dated addendum — this script only measures
and records, it doesn't editorialize.

Usage:
    python scripts/architecture_ablation.py --n 20 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepresearch.agent.orchestrator import run_research  # noqa: E402
from deepresearch.backends.local_corpus import LocalCorpusBackend  # noqa: E402
from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.store import db  # noqa: E402
from deepresearch.telemetry.otel_setup import init_telemetry  # noqa: E402

from eval.benchmarks import musique as musique_bench  # noqa: E402
from eval.metrics.answer_f1 import best_answer_f1, gold_contained  # noqa: E402
from eval.metrics.trajectory import compute_trajectory_metrics  # noqa: E402
from eval.run_eval import make_llm  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"

VARIANTS = [
    ("plan_first_pool4", {"planning_style": "plan_first", "max_workers": 4}),
    ("plan_first_pool1", {"planning_style": "plan_first", "max_workers": 1}),
    ("react", {"planning_style": "react", "max_react_steps": 4}),
]


async def _run_variant(examples, *, database_url, llm, variant_name, config_overrides) -> dict:
    results = []
    scores_rows = []
    tool_calls_by_run: dict[str, list[dict]] = {}
    start = time.monotonic()

    for ex in examples:
        config = RunConfig(
            database_url=database_url,
            search_backend="local_corpus",
            local_corpus_dir=str(ex.corpus_path),
            cache_enabled=False,
            **config_overrides,
        )
        backend = LocalCorpusBackend.from_json_file(ex.corpus_path)
        result = await run_research(
            ex.question, config=config, search_backend=backend, llm=llm, benchmark_name=f"ablation_{variant_name}"
        )
        results.append(result)

        predicted = result.report.text if result.report else ""
        f1 = best_answer_f1(predicted, ex.gold_answers)
        contains_gold = float(any(gold_contained(predicted, g) for g in ex.gold_answers))
        scores_rows.append(
            {
                "run_id": result.run_id,
                "benchmark_name": f"ablation_{variant_name}",
                "question_id": ex.question_id,
                "metric_name": "answer_f1",
                "value": f1,
                "judge_model": None,
                "rubric_version": None,
                "raw_judge_output": None,
            }
        )
        scores_rows.append(
            {
                "run_id": result.run_id,
                "benchmark_name": f"ablation_{variant_name}",
                "question_id": ex.question_id,
                "metric_name": "answer_contains_gold",
                "value": contains_gold,
                "judge_model": None,
                "rubric_version": None,
                "raw_judge_output": None,
            }
        )
        tool_calls_by_run[result.run_id] = await db.get_tool_calls_for_run(database_url, result.run_id)

    elapsed = time.monotonic() - start
    traj = compute_trajectory_metrics(results, tool_calls_by_run)
    await db.bulk_insert_eval_scores(database_url, scores_rows)

    n = len(examples)
    f1_values = [r["value"] for r in scores_rows if r["metric_name"] == "answer_f1"]
    contains_values = [r["value"] for r in scores_rows if r["metric_name"] == "answer_contains_gold"]

    return {
        "variant": variant_name,
        "config_overrides": config_overrides,
        "n": n,
        "wall_clock_seconds_total": elapsed,
        "wall_clock_seconds_per_question": elapsed / n if n else 0.0,
        "mean_answer_f1": sum(f1_values) / n if n else 0.0,
        "mean_answer_contains_gold": sum(contains_values) / n if n else 0.0,
        "total_agent_cost_usd": sum(r.total_cost_usd for r in results),
        "mean_iterations_per_question": sum(r.iterations for r in results) / n if n else 0.0,
        "trajectory": traj.summary(),
    }


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-url", type=str, default=None)
    args = parser.parse_args()

    init_telemetry()
    database_url = args.database_url or RunConfig().database_url
    await db.ensure_schema(database_url)

    examples = musique_bench.load_subset(args.n, args.seed)
    llm, is_real = make_llm()
    print(
        f"[ablation] {len(examples)} questions x {len(VARIANTS)} variants = "
        f"{len(examples) * len(VARIANTS)} runs. is_real_llm={is_real}"
    )

    summaries = []
    for name, overrides in VARIANTS:
        print(f"\n--- variant: {name} ({overrides}) ---")
        summary = await _run_variant(
            examples, database_url=database_url, llm=llm, variant_name=name, config_overrides=overrides
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, default=str))

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"is_real_llm": is_real, "n": args.n, "seed": args.seed, "variants": summaries}
    out_path = RESULTS_DIR / f"architecture_ablation_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
