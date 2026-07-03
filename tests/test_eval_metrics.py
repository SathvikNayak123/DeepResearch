from __future__ import annotations

import pytest

from eval.metrics.answer_f1 import best_answer_f1, gold_contained, token_f1
from eval.metrics.reliability import compute_reliability
from eval.metrics.trajectory import compute_trajectory_metrics
from deepresearch.schemas import Plan, RunResult, RunStatus


def test_token_f1_exact_match_is_one():
    assert token_f1("Paris", "Paris") == 1.0


def test_token_f1_partial_overlap():
    f1 = token_f1("the capital city of France", "capital of France")
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("bananas", "Paris") == 0.0


def test_best_answer_f1_picks_best_alias():
    assert best_answer_f1("Bill", ["William", "Bill", "Billy"]) == 1.0


def test_gold_contained_case_and_punctuation_insensitive():
    assert gold_contained("The answer is: PARIS, obviously.", "paris") is True
    assert gold_contained("The answer is London.", "paris") is False


def test_compute_reliability_all_consistent():
    per_question = {"q1": [True, True, True], "q2": [False, False, False]}
    report = compute_reliability(per_question)
    assert report.all_consistent_rate == 1.0
    assert report.mean_accuracy == 0.5  # q1 always right, q2 always wrong -> 50% per repeat


def test_compute_reliability_inconsistent_question_lowers_rate():
    per_question = {"q1": [True, True, True], "q2": [True, False, True]}
    report = compute_reliability(per_question)
    assert report.all_consistent_rate == 0.5  # only q1 is fully consistent
    assert report.stdev_accuracy >= 0.0


def _make_run_result(run_id: str, status: RunStatus, tokens: int = 100) -> RunResult:
    return RunResult(
        run_id=run_id,
        status=status,
        question="q",
        plan=Plan(sub_questions=[]),
        worker_notes=[],
        reflections=[],
        total_tokens_in=tokens,
        total_tokens_out=tokens,
    )


def test_trajectory_metrics_completion_and_tool_success():
    results = [
        _make_run_result("r1", RunStatus.COMPLETED, tokens=50),
        _make_run_result("r2", RunStatus.BUDGET_EXCEEDED, tokens=999),
    ]
    tool_calls_by_run = {
        "r1": [{"success": True}, {"success": True}],
        "r2": [{"success": False}],
    }
    metrics = compute_trajectory_metrics(results, tool_calls_by_run)
    assert metrics.task_completion_rate == 0.5
    assert metrics.tool_call_success_rate == pytest.approx(2 / 3)
    assert metrics.mean_steps_per_solved_task == 2.0  # only r1 (completed) counts
    assert metrics.mean_tokens_per_solved_task == 100.0  # r1's tokens_in + tokens_out
