"""RACE (Referee-based Adaptive Criteria-driven Evaluation) — DeepResearch
Bench's report-quality judge, adapted from `Ayanami0730/deep_research_bench`
(MIT licensed), commit 469cce54ea7f6a63c163d3d9fec879cf289ec484.

This is the point-wise (single-article, no reference-article comparison)
variant of their scoring: for each of the task's pre-generated, officially
weighted criteria (`data/criteria_data/criteria.jsonl` in their repo, fetched
and cached by eval/benchmarks/deepresearch_bench.py), a judge model scores
our agent's report 0-10, then scores are aggregated into 4 dimension scores
and one overall score via the exact weighted-average math their
`utils/score_calculator.py` uses (target-only branch - we have one article to
score, not two to compare).

Judge model is whatever RunConfig.judge_model is set to for this run, not
necessarily the reference implementation's GPT-5.5 (see
docs/RESULTS.md for what model was actually used and why) - scores from this
module are therefore not directly comparable to the public DRB leaderboard,
which is scored under a fixed evaluator model.
"""

from __future__ import annotations

import json

from deepresearch.llm.client import LLMClient, LLMUsage

from eval.prompts.loader import load_eval_prompt

DIMENSIONS = ["comprehensiveness", "insight", "instruction_following", "readability"]

_CRITERION_SCORE_ITEM = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "analysis": {"type": "string"},
        "target_score": {"type": "number"},
    },
    "required": ["criterion", "analysis", "target_score"],
    "additionalProperties": False,
}

RACE_SCHEMA = {
    "type": "object",
    "properties": {dim: {"type": "array", "items": _CRITERION_SCORE_ITEM} for dim in DIMENSIONS},
    "required": DIMENSIONS,
    "additionalProperties": False,
}


async def score_report(
    llm: LLMClient, *, model: str, task_prompt: str, article: str, criteria_data: dict
) -> tuple[dict, LLMUsage]:
    """One judge call, scoring `article` against every criterion in
    criteria_data["criterions"]. Returns the raw per-criterion LLM output —
    call aggregate_race_score() on the result to get dimension/overall
    scores."""
    system = load_eval_prompt("race_pointwise_v1.txt")
    criteria_json = json.dumps(criteria_data["criterions"], indent=2)
    user_content = (
        f"Task:\n{task_prompt}\n\n"
        f"Article to evaluate:\n{article}\n\n"
        f"Evaluation criteria (JSON, grouped by dimension):\n{criteria_json}"
    )
    data, usage = await llm.complete_json(
        model=model,
        system=system,
        user_content=user_content,
        schema=RACE_SCHEMA,
        max_tokens=8192,
    )
    return data, usage


def aggregate_race_score(llm_output: dict, criteria_data: dict) -> dict:
    """Weighted-average aggregation - a direct port of
    Ayanami0730/deep_research_bench's utils/score_calculator.py
    calculate_weighted_scores(), target-only branch (no reference article).

    Per dimension: weighted_avg = sum(score_i * weight_i) / sum(weight_i),
    matched by criterion text (case-insensitive, substring-fallback, same as
    the reference implementation - a judge model won't always echo criterion
    text byte-for-byte). Overall: sum(dimension_weighted_avg * dimension_weight).
    """
    dimension_weights: dict = criteria_data.get("dimension_weight", {})
    criterion_weights: dict[str, dict[str, float]] = {
        dim: {c["criterion"]: c["weight"] for c in criterions}
        for dim, criterions in criteria_data.get("criterions", {}).items()
    }

    dims_out: dict[str, float] = {}
    total = 0.0

    for dim, scores_list in llm_output.items():
        weights_by_criterion = criterion_weights.get(dim)
        if not isinstance(scores_list, list) or not weights_by_criterion:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        for item in scores_list:
            criterion_text = (item.get("criterion") or "").strip()
            score = item.get("target_score")
            if not criterion_text or score is None:
                continue

            weight = _match_weight(criterion_text, weights_by_criterion)
            weighted_sum += float(score) * weight
            total_weight += weight

        dim_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        dims_out[dim] = dim_avg
        total += dim_avg * dimension_weights.get(dim, 0.0)

    return {**dims_out, "total": total}


def _match_weight(criterion_text: str, weights_by_criterion: dict[str, float]) -> float:
    if criterion_text in weights_by_criterion:
        return weights_by_criterion[criterion_text]

    lowered = criterion_text.lower()
    for key, weight in weights_by_criterion.items():
        if key.lower() == lowered:
            return weight
    for key, weight in weights_by_criterion.items():
        if lowered in key.lower() or key.lower() in lowered:
            return weight

    # No match at all (a judge invented/rephrased a criterion beyond
    # recognition) - fall back to the dimension's average weight, same as
    # the reference implementation, rather than dropping the score.
    return sum(weights_by_criterion.values()) / len(weights_by_criterion)
