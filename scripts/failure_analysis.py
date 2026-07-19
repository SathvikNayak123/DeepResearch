"""Failure-mode classifier for FRAMES / MuSiQue runs (docs/DESIGN.md §7).

Requirement-2 work ("improve the scores") is gated by the project's own rule:
don't build a score lever before failure-mode data says which failure it
addresses. This script produces that data. For every *missed* benchmark
question in a run store, it decides whether the miss is a:

  - retrieval_or_composition : the gold answer never appeared in the
      reranked chunks the workers were given. Either the retriever/reranker
      failed to surface the evidence, OR (for multi-hop / computed answers)
      the gold never appears verbatim in any single source and the real gap
      is reasoning — this bucket deliberately conflates the two rather than
      overclaiming pure retrieval (the per-question JSON keeps gold_in_chunks
      so a spot-check can separate them). Points at rerank_top_k /
      max_chunks_per_source / candidate_pool_size / graph-retrieval levers.
  - extraction : the gold WAS in a chunk a worker saw, but no worker claim
      captured it. Points at worker-prompt / model-tier levers.
  - synthesis : a worker claim captured the gold, but the final report/answer
      is still wrong. Points at synthesis-prompt / answer-first / coverage-gate
      levers.

Read-only: no DB writes, no LLM calls (the "missed" verdict reuses stored
judge scores when present, else the gold_contained string proxy; the
retrieval re-run is BM25 + reranker only). The retrieval re-run uses each
run's OWN stored config (runs.config), not today's defaults, so attribution
matches the run that produced the miss.

Honest limitations, stated up front:
  - The retrieval re-run is a best-effort reconstruction, not a byte-exact
    replay: reranker/embedding nondeterminism means the selected chunks can
    differ slightly from what the original run actually saw.
  - gold_contained is a normalized-substring proxy: an answer a judge would
    accept but that doesn't contain the gold string verbatim (a paraphrase,
    an alias not in the alias list) reads as a miss here. Stored judge scores
    are preferred when available to blunt this.

Usage:
    python scripts/failure_analysis.py --database-url sqlite+aiosqlite:///./sanity_check.db
    python scripts/failure_analysis.py --database-url "$DBURL" --benchmark frames --n 20 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # so `import eval.*` resolves when run as a script

from deepresearch.backends.local_corpus import LocalCorpusBackend  # noqa: E402
from deepresearch.chunking import cap_chunks, chunk_text  # noqa: E402
from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.rerank import build_rerank_backend  # noqa: E402
from deepresearch.store import db  # noqa: E402
from deepresearch.store.models import eval_scores, runs  # noqa: E402
from sqlalchemy import select  # noqa: E402

from eval.benchmarks import frames as frames_bench  # noqa: E402
from eval.benchmarks import musique as musique_bench  # noqa: E402
from eval.metrics.answer_f1 import gold_contained  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"

# eval_scores metric that carries the "was this question correct" signal per
# benchmark — preferred over the gold_contained proxy when a row exists.
_CORRECTNESS_METRIC = {"frames": "accuracy", "musique": "answer_contains_gold"}


def _norm_q(question: str) -> str:
    return " ".join(question.lower().split())


async def _load_gold_index(benchmark: str, n: int, seed: int) -> dict[str, dict]:
    """Deterministic subset -> {normalized_question: {answers, corpus_path,
    tag}}. Uses the same samplers/seed the eval harness used, so the run's
    questions match by string."""
    index: dict[str, dict] = {}
    if benchmark == "musique":
        for ex in musique_bench.load_subset(n, seed):
            index[_norm_q(ex.question)] = {
                "answers": ex.gold_answers,
                "corpus_path": ex.corpus_path,
                "tag": f"{ex.hop_count}hop",
            }
    else:  # frames
        rows = frames_bench.load_subset(n, seed)
        # ingest_corpus is a no-op fetch when the corpus JSON is already
        # cached on disk (it is, for any subset that has actually been run) —
        # so this stays offline in practice.
        for ex in await frames_bench.ingest_corpus(rows):
            index[_norm_q(ex.question)] = {
                "answers": [ex.answer],
                "corpus_path": ex.corpus_path,
                "tag": ex.reasoning_type,
            }
    return index


def _recover_run(trajectories: list[dict]) -> dict:
    """Pull question / sub-questions / report / claims back out of the stored
    trajectory rows — no re-planning, no LLM. Works for both plan-first and
    react runs (synthesis always carries the question; workers always carry
    their sub_question)."""
    question = None
    sub_questions: list[str] = []
    report_text = ""
    claims: list[dict] = []

    for t in trajectories:
        stage = t.get("stage")
        inp = t.get("input") or {}
        out = t.get("output") or {}
        if stage == "synthesis" and inp.get("question"):
            question = inp["question"]
        elif stage == "plan" and inp.get("question") and question is None:
            question = inp["question"]
        if stage == "worker":
            sq = inp.get("sub_question")
            if sq:
                sub_questions.append(sq)
            for c in out.get("claims", []):
                claims.append(c)
        if stage == "synthesis":
            report_text = out.get("text", "") or ""

    return {
        "question": question,
        "sub_questions": sub_questions,
        "report_text": report_text,
        "claims": claims,
    }


def _gold_in_texts(golds: list[str], texts: list[str]) -> bool:
    return any(gold_contained(text, g) for text in texts for g in golds)


# The reranker (bge-reranker-v2-m3, ~600MB) loads lazily on first use and is
# reusable across questions — a fresh backend per question would reload the
# model from disk every time (the difference between ~1min and >20min for a
# 20-question analysis). All runs in one eval DB share the same rerank config,
# so cache the instance keyed on the config knobs that actually change its
# behavior.
_RERANK_CACHE: dict[tuple, object] = {}


def _get_rerank_backend(config: RunConfig):
    key = (config.rerank_enabled, config.rerank_backend)
    if key not in _RERANK_CACHE:
        _RERANK_CACHE[key] = build_rerank_backend(config)
    return _RERANK_CACHE[key]


async def _selected_chunks_for(sub_questions: list[str], corpus_path: Path, config: RunConfig) -> list[str]:
    """Read-only replica of worker.run_worker's retrieval half (search ->
    fetch -> cap_chunks(chunk_text) -> rerank -> top_k), across every
    sub-question the run used. Deliberately NOT calling run_worker (which
    also runs the LLM, mutates a source registry, and records tool_calls) —
    this is retrieval only, so it stays free and side-effect-free. Kept in
    sync with worker.py by using the same public helpers/config knobs."""
    backend = LocalCorpusBackend.from_json_file(corpus_path)
    rerank_backend = _get_rerank_backend(config)
    selected: list[str] = []

    for sq in sub_questions:
        try:
            results = await backend.search(sq, max_results=config.candidate_pool_size)
        except Exception:
            continue
        candidates: list[str] = []
        for r in results:
            try:
                content = (await backend.fetch(r.url)).content or r.snippet
            except Exception:
                content = r.snippet
            for chunk in cap_chunks(chunk_text(content) or [content], config.max_chunks_per_source):
                candidates.append(chunk)
        if config.rerank_enabled and rerank_backend is not None and candidates:
            ranked = await rerank_backend.rerank(sq, candidates)
            selected.extend(candidates[rc.index] for rc in ranked[: config.rerank_top_k])
        else:
            selected.extend(candidates[: config.rerank_top_k])
    return selected


async def _correct_signal(database_url: str, run_id: str, benchmark: str, golds: list[str], report_text: str) -> bool:
    """Stored judge/contains_gold score if present (faithful to the real
    metric), else the gold_contained proxy over the report text."""
    metric = _CORRECTNESS_METRIC[benchmark]
    engine = db.get_engine(database_url)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                select(eval_scores.c.value).where(
                    eval_scores.c.run_id == run_id, eval_scores.c.metric_name == metric
                )
            )
        ).fetchall()
    if rows:
        return float(rows[0][0]) >= 0.5
    return _gold_in_texts(golds, [report_text])


async def analyze(database_url: str, benchmark: str, n: int, seed: int, limit: int | None, rerun: bool) -> dict:
    gold_index = await _load_gold_index(benchmark, n, seed)

    engine = db.get_engine(database_url)
    async with engine.begin() as conn:
        run_rows = (
            await conn.execute(
                select(runs.c.run_id, runs.c.config, runs.c.created_at)
                .where(runs.c.benchmark_name == benchmark, runs.c.status == "completed")
                .order_by(runs.c.created_at)
            )
        ).fetchall()

    per_question: list[dict] = []
    analyzed = 0
    for run_row in run_rows:
        if limit is not None and analyzed >= limit:
            break
        run_id = run_row[0]
        run_config = run_row[1] if isinstance(run_row[1], dict) else json.loads(run_row[1])

        trajectories = await db.get_trajectories_for_run(database_url, run_id)
        recovered = _recover_run(trajectories)
        q = recovered["question"]
        if q is None:
            continue
        gold = gold_index.get(_norm_q(q))
        if gold is None:
            continue  # a run whose question isn't in this subset (different seed/n)
        analyzed += 1
        golds = gold["answers"]

        correct = await _correct_signal(database_url, run_id, benchmark, golds, recovered["report_text"])
        record = {
            "run_id": run_id,
            "question": q,
            "tag": gold["tag"],
            "gold": golds[0] if golds else "",
            "correct": correct,
            "gold_in_report": _gold_in_texts(golds, [recovered["report_text"]]),
            "gold_in_claims": _gold_in_texts(
                golds, [c.get("text", "") for c in recovered["claims"]] + [c.get("quote", "") for c in recovered["claims"]]
            ),
        }

        if correct:
            record["failure_class"] = "correct"
        else:
            gold_in_chunks = None
            if rerun:
                config = RunConfig.from_overrides(run_config)
                # The bge reranker re-run is ~30s/sub-question on CPU — print
                # per-miss progress so a multi-minute analysis isn't an opaque
                # black box (pass --no-rerun-retrieval to skip it entirely).
                print(f"  [{benchmark}] classifying miss {analyzed}: {q[:60]}...", flush=True)
                chunks = await _selected_chunks_for(recovered["sub_questions"], gold["corpus_path"], config)
                gold_in_chunks = _gold_in_texts(golds, chunks)
            record["gold_in_chunks"] = gold_in_chunks

            if rerun and gold_in_chunks is False:
                record["failure_class"] = "retrieval_or_composition"
            elif record["gold_in_claims"]:
                record["failure_class"] = "synthesis"
            elif rerun and gold_in_chunks:
                record["failure_class"] = "extraction"
            else:
                # rerun disabled, or gold not in claims and chunks unknown —
                # can't separate retrieval from extraction without the re-run.
                record["failure_class"] = "unclassified_miss"
        per_question.append(record)

    dist = Counter(r["failure_class"] for r in per_question)
    n_missed = sum(1 for r in per_question if not r["correct"])
    return {
        "benchmark": benchmark,
        "database_url": database_url,
        "n_analyzed": len(per_question),
        "n_correct": sum(1 for r in per_question if r["correct"]),
        "n_missed": n_missed,
        "distribution": dict(dist),
        "per_question": per_question,
    }


def _print_summary(summary: dict) -> None:
    b = summary["benchmark"]
    print(f"\n=== {b}: {summary['n_analyzed']} analyzed, "
          f"{summary['n_correct']} correct, {summary['n_missed']} missed ===")
    miss_classes = {k: v for k, v in summary["distribution"].items() if k != "correct"}
    total_missed = summary["n_missed"] or 1
    for cls in ["retrieval_or_composition", "extraction", "synthesis", "unclassified_miss"]:
        if cls in miss_classes:
            v = miss_classes[cls]
            print(f"  {cls:26} {v:3}  ({v / total_missed * 100:.0f}% of misses)")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--benchmark", choices=["frames", "musique", "both"], default="both")
    parser.add_argument("--n", type=int, default=20, help="subset size the runs were sampled from (match the eval run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap questions analyzed (quick smoke)")
    parser.add_argument("--no-rerun-retrieval", action="store_true",
                        help="skip the BM25+rerank re-run (faster, but misses can't be split retrieval-vs-extraction)")
    args = parser.parse_args()

    benchmarks = ["frames", "musique"] if args.benchmark == "both" else [args.benchmark]
    rerun = not args.no_rerun_retrieval

    summaries = {}
    for b in benchmarks:
        summary = await analyze(args.database_url, b, args.n, args.seed, args.limit, rerun)
        summaries[b] = summary
        _print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"failure_analysis_{ts}.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
