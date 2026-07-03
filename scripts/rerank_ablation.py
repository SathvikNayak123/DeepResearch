"""Standalone rerank ablation: raw retrieval order vs. cross-encoder reranked
order, on MuSiQue questions with known gold ("is_supporting") paragraphs.

No LLM judge, no full agent loop — this measures the rerank stage in
isolation against ground truth, per this session's brief. Draws real
questions + candidate pools + gold labels from the MuSiQue mirror
(bdsaglam/musique, "answerable" config, CC BY 4.0), which ships each
question's gold + distractor paragraphs together — exactly the retrieval
ablation setup docs/DESIGN.md commits to in section 5.4.

Usage:
    python scripts/rerank_ablation.py [--n 50] [--seed 42] [--k 3 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepresearch.rerank.bge import DEFAULT_MODEL, CrossEncoderRerankBackend  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET = "bdsaglam/musique"
DATASET_CONFIG = "answerable"
DATASET_SPLIT = "validation"


@dataclass
class QuestionSample:
    id: str
    question: str
    paragraphs: list[str]
    relevant_positions: set[int]


def load_sample(n: int, seed: int) -> list[QuestionSample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    import random

    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)

    samples: list[QuestionSample] = []
    for i in idxs:
        row = ds[i]
        paragraphs = row["paragraphs"]
        relevant = {pos for pos, p in enumerate(paragraphs) if p["is_supporting"]}
        if not relevant:
            continue
        samples.append(
            QuestionSample(
                id=row["id"],
                question=row["question"],
                paragraphs=[p["paragraph_text"] for p in paragraphs],
                relevant_positions=relevant,
            )
        )
        if len(samples) >= n:
            break
    return samples


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg_at_k(ranked_relevance: list[int], n_relevant: int, k: int) -> float:
    ideal = [1] * min(n_relevant, k) + [0] * max(0, k - n_relevant)
    ideal_dcg = _dcg(ideal, k)
    return _dcg(ranked_relevance, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def metrics_for_order(order: list[int], relevant: set[int], k_values: list[int]) -> dict[str, float]:
    n_relevant = len(relevant)
    ranked_relevance = [1 if pos in relevant else 0 for pos in order]
    out: dict[str, float] = {}
    for k in k_values:
        top_k = order[:k]
        out[f"hit_rate@{k}"] = 1.0 if any(pos in relevant for pos in top_k) else 0.0
        out[f"recall@{k}"] = len(set(top_k) & relevant) / n_relevant if n_relevant else 0.0
        out[f"ndcg@{k}"] = _ndcg_at_k(ranked_relevance, n_relevant, k)
    return out


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


async def run_ablation(n: int, seed: int, k_values: list[int], model_name: str) -> dict:
    samples = load_sample(n, seed)
    backend = CrossEncoderRerankBackend(model_name=model_name)

    # Warm up: load the model + JIT/first-call overhead outside the timed loop.
    await backend.rerank("warmup query", ["warmup document about nothing in particular."])

    raw_metrics: list[dict[str, float]] = []
    reranked_metrics: list[dict[str, float]] = []
    latencies_ms: list[float] = []

    for sample in samples:
        raw_order = list(range(len(sample.paragraphs)))
        raw_metrics.append(metrics_for_order(raw_order, sample.relevant_positions, k_values))

        start = time.perf_counter()
        ranked = await backend.rerank(sample.question, sample.paragraphs)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        reranked_order = [rc.index for rc in ranked]
        reranked_metrics.append(metrics_for_order(reranked_order, sample.relevant_positions, k_values))

    def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = rows[0].keys() if rows else []
        return {k: _mean([r[k] for r in rows]) for k in keys}

    latencies_sorted = sorted(latencies_ms)

    def pctl(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        idx = min(int(len(latencies_sorted) * p), len(latencies_sorted) - 1)
        return latencies_sorted[idx]

    return {
        "config": {
            "dataset": DATASET,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "n_questions": len(samples),
            "seed": seed,
            "k_values": k_values,
            "rerank_model": model_name,
        },
        "raw": aggregate(raw_metrics),
        "reranked": aggregate(reranked_metrics),
        "delta": {
            k: aggregate(reranked_metrics)[k] - aggregate(raw_metrics)[k] for k in aggregate(raw_metrics)
        },
        "latency_ms": {
            "mean": _mean(latencies_ms),
            "p50": pctl(0.50),
            "p95": pctl(0.95),
            "n": len(latencies_ms),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = asyncio.run(run_ablation(args.n, args.seed, args.k, args.model))

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"rerank_ablation_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
