from __future__ import annotations

import pytest

from deepresearch.store import db
from eval.ci_baseline import (
    check_regression,
    compute_current_metrics,
    load_baseline,
    render_gate_table,
    write_baseline,
)


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'ci_baseline_test.db'}"


@pytest.mark.asyncio
async def test_compute_current_metrics_averages_across_runs(db_url):
    await db.init_schema(db_url)
    for i, (accuracy, cost) in enumerate([(1.0, 0.10), (0.0, 0.20)]):
        run_id = f"{'a' * 31}{i}"
        await db.create_run(db_url, run_id=run_id, benchmark_name="frames", config={}, git_sha="abc", status="running")
        await db.finish_run(db_url, run_id=run_id, status="completed", total_cost_usd=cost, total_latency_ms=100)
        await db.bulk_insert_eval_scores(
            db_url,
            [
                {
                    "run_id": run_id,
                    "benchmark_name": "frames",
                    "question_id": f"q{i}",
                    "metric_name": "accuracy",
                    "value": accuracy,
                    "judge_model": None,
                    "rubric_version": None,
                    "raw_judge_output": None,
                }
            ],
        )

    metrics = await compute_current_metrics(db_url)
    assert metrics["frames.accuracy"] == pytest.approx(0.5)
    assert metrics["frames.cost_per_query_usd"] == pytest.approx(0.15)
    assert metrics["frames.task_completion_rate"] == pytest.approx(1.0)  # both fixture runs completed
    assert "musique.answer_f1" not in metrics  # no musique rows inserted
    assert "musique.task_completion_rate" not in metrics


@pytest.mark.asyncio
async def test_compute_current_metrics_task_completion_rate_with_mixed_statuses(db_url):
    await db.init_schema(db_url)
    statuses = ["completed", "completed", "completed", "budget_exceeded"]
    for i, status in enumerate(statuses):
        run_id = f"{'c' * 31}{i}"
        await db.create_run(db_url, run_id=run_id, benchmark_name="musique", config={}, git_sha="abc", status="running")
        await db.finish_run(db_url, run_id=run_id, status=status, total_cost_usd=0.0, total_latency_ms=100)

    metrics = await compute_current_metrics(db_url)
    assert metrics["musique.task_completion_rate"] == pytest.approx(0.75)


def test_write_and_load_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(path, {"frames.accuracy": 0.7}, config={"seeded_from": "test"})

    loaded = load_baseline(path)
    assert loaded["metrics"]["frames.accuracy"] == 0.7
    assert loaded["config"]["seeded_from"] == "test"
    assert "git_sha" in loaded


def test_load_baseline_missing_file_returns_none(tmp_path):
    assert load_baseline(tmp_path / "does_not_exist.json") is None


@pytest.mark.parametrize(
    "key,baseline_value,current_value,expect_regression",
    [
        ("frames.accuracy", 0.70, 0.68, False),  # 2pt drop, within 3pt tolerance
        ("frames.accuracy", 0.70, 0.66, True),  # 4pt drop, over tolerance
        ("frames.citation_precision", 0.70, 0.70, False),  # no change
        ("musique.cost_per_query_usd", 0.10, 0.12, False),  # +20%, within 25% tolerance
        ("musique.cost_per_query_usd", 0.10, 0.13, True),  # +30%, over tolerance
        ("frames.cost_per_query_usd", 0.10, 0.05, False),  # cost going down never regresses
    ],
)
def test_check_regression_tolerances(key, baseline_value, current_value, expect_regression):
    reason = check_regression(key, baseline_value, current_value)
    assert (reason is not None) == expect_regression


def test_render_gate_table_reports_failures_and_skips_missing_metrics():
    baseline = {
        "frames.task_completion_rate": 1.0,
        "frames.citation_precision": 0.70,
        "musique.answer_f1": 0.02,
    }
    current = {
        "frames.task_completion_rate": 0.90,  # gated metric, regressed
        "frames.citation_precision": 0.70,
    }  # musique.answer_f1 missing from current

    table, failures = render_gate_table(baseline, current)

    assert len(failures) == 1
    assert "frames.task_completion_rate" in failures[0]
    assert "SKIPPED" in table  # musique.answer_f1 present only in baseline


def test_render_gate_table_does_not_fail_on_informational_only_regression():
    """accuracy/citation-precision/answer_f1* are measured and shown every
    PR but don't gate PR-smoke — see INFORMATIONAL_ONLY_METRICS's comment
    for why a flat tolerance can't survive their measured single-run noise."""
    baseline = {"frames.accuracy": 0.70, "musique.answer_f1_extracted": 0.50}
    current = {"frames.accuracy": 0.40, "musique.answer_f1_extracted": 0.20}  # both regress hard

    table, failures = render_gate_table(baseline, current)

    assert failures == []
    assert "INFO (not gated on PR-smoke)" in table
    assert "FAIL" not in table


def test_render_gate_table_still_fails_on_gated_cost_regression():
    baseline = {"musique.cost_per_query_usd": 0.10}
    current = {"musique.cost_per_query_usd": 0.20}  # +100%, over the 25% tolerance

    table, failures = render_gate_table(baseline, current)

    assert len(failures) == 1
    assert "musique.cost_per_query_usd" in failures[0]
