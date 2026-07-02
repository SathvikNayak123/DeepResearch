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
    assert "musique.answer_f1" not in metrics  # no musique rows inserted


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
    baseline = {"frames.accuracy": 0.70, "frames.citation_precision": 0.70, "musique.answer_f1": 0.02}
    current = {"frames.accuracy": 0.60, "frames.citation_precision": 0.70}  # accuracy regressed, musique missing

    table, failures = render_gate_table(baseline, current)

    assert len(failures) == 1
    assert "frames.accuracy" in failures[0]
    assert "SKIPPED" in table  # musique.answer_f1 present only in baseline
