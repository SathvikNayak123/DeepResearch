from __future__ import annotations

import pytest

from eval.metrics.answer_f1 import best_answer_f1, gold_contained, token_f1
from eval.metrics.citation import compute_citation_metrics
from eval.metrics.reliability import compute_reliability
from eval.metrics.trajectory import compute_trajectory_metrics
from deepresearch.schemas import Claim, Plan, Report, RunResult, RunStatus, SourceRegistryEntry, WorkerNotes


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


class _StubJudge:
    """Minimal Judge double for the coverage path — precision needs an LLM
    verdict, but coverage (the metric the regex bug zeroed out) is pure
    marker-vs-claim bookkeeping. Always-supported keeps precision defined."""

    async def judge_citation(self, *, claim: str, quote: str) -> tuple[bool, str, bool]:
        return True, "stub", False


def _entry(source_id: str) -> SourceRegistryEntry:
    return SourceRegistryEntry(source_id=source_id, url=f"https://example.com/{source_id}", title=source_id)


@pytest.mark.asyncio
async def test_compute_citation_metrics_counts_namespaced_ids_as_covered():
    """A report citing a namespaced id (`src_1abe120f_1`, LangGraph's
    per-node registry namespacing) must count the matching claim as covered,
    not silently drop it to 0."""
    notes = [
        WorkerNotes(
            sub_question_id="1abe120f",
            sub_question="capital of France",
            claims=[
                Claim(text="Paris is the capital", source_id="src_1abe120f_1", quote="Paris ...", confidence=0.9)
            ],
        )
    ]
    report = Report(text="Paris is the capital of France [src_1abe120f_1].", citations=[_entry("src_1abe120f_1")])
    result = RunResult(
        run_id="r", status=RunStatus.COMPLETED, question="q",
        plan=Plan(sub_questions=[]), worker_notes=notes, reflections=[], report=report,
    )
    metrics = await compute_citation_metrics(result, _StubJudge())
    assert metrics.coverage == 1.0
    assert metrics.precision == 1.0
    assert metrics.n_claims_checked == 1


@pytest.mark.asyncio
async def test_compute_citation_metrics_counts_both_ids_in_one_bracket():
    """Regression lock for a real bug: a regex-based version of this metric
    (`\\[(src_\\w+)\\]`) silently dropped a whole citation whenever the model
    wrote two ids in one bracket (`[src_a, src_b]`) -- `\\w+` can't span the
    comma, so the marker never matched at all, undercounting coverage with no
    error. Reading Report.citations (the already-validated structured
    synthesis output) instead of the free-form text must count both claims as
    covered regardless of how the model punctuated the citation in prose."""
    notes = [
        WorkerNotes(
            sub_question_id="n1",
            sub_question="who is the lead singer",
            claims=[
                Claim(text="fronts Ratata", source_id="src_n1_r0_1", quote="fronted by Mauro Scocco", confidence=0.9),
                Claim(text="Swedish group", source_id="src_n1_r0_4", quote="Swedish pop group", confidence=0.9),
            ],
        ),
        WorkerNotes(
            sub_question_id="n2",
            sub_question="birth year",
            claims=[Claim(text="born 1962", source_id="src_n2_r0_1", quote="born in 1962", confidence=0.9)],
        ),
    ]
    report = Report(
        text=(
            "The lead singer of the Swedish pop group Ratata, Mauro Scocco "
            "[src_n1_r0_1, src_n1_r0_4], was born in 1962 [src_n2_r0_1]."
        ),
        citations=[_entry("src_n1_r0_1"), _entry("src_n1_r0_4"), _entry("src_n2_r0_1")],
    )
    result = RunResult(
        run_id="r", status=RunStatus.COMPLETED, question="q",
        plan=Plan(sub_questions=[]), worker_notes=notes, reflections=[], report=report,
    )
    metrics = await compute_citation_metrics(result, _StubJudge())
    assert metrics.coverage == 1.0  # all 3 claims covered, not just the one a regex would have found
    assert metrics.n_claims_checked == 3


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
