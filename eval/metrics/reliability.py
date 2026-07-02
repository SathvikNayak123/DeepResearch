"""Reliability job — docs/DESIGN.md §5.2: a fixed subset repeated 3-5x,
reported as a *distribution* (variance across repeats) and an
all-runs-consistent rate (pass^k-style), never a single point estimate.
CLAUDE.md: "an accuracy figure without it is incomplete and should not be
cited on its own."
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class ReliabilityReport:
    n_questions: int
    n_repeats: int
    per_question_correct: dict[str, list[bool]]  # question_id -> [correct? per repeat]
    per_repeat_accuracy: list[float]
    mean_accuracy: float
    stdev_accuracy: float
    all_consistent_rate: float  # fraction of questions where every repeat agreed (all-correct or all-incorrect)

    def summary(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "n_repeats": self.n_repeats,
            "per_repeat_accuracy": self.per_repeat_accuracy,
            "mean_accuracy": self.mean_accuracy,
            "stdev_accuracy": self.stdev_accuracy,
            "all_consistent_rate": self.all_consistent_rate,
        }


def compute_reliability(per_question_correct: dict[str, list[bool]]) -> ReliabilityReport:
    n_repeats = max((len(v) for v in per_question_correct.values()), default=0)
    per_repeat_accuracy = []
    for r in range(n_repeats):
        vals = [v[r] for v in per_question_correct.values() if len(v) > r]
        per_repeat_accuracy.append(sum(vals) / len(vals) if vals else 0.0)

    n_questions = len(per_question_correct)
    all_consistent = sum(1 for v in per_question_correct.values() if len(set(v)) == 1)

    return ReliabilityReport(
        n_questions=n_questions,
        n_repeats=n_repeats,
        per_question_correct=per_question_correct,
        per_repeat_accuracy=per_repeat_accuracy,
        mean_accuracy=statistics.fmean(per_repeat_accuracy) if per_repeat_accuracy else 0.0,
        stdev_accuracy=statistics.pstdev(per_repeat_accuracy) if len(per_repeat_accuracy) > 1 else 0.0,
        all_consistent_rate=(all_consistent / n_questions) if n_questions else 0.0,
    )
