"""Agentic metrics — docs/DESIGN.md §5.2: task completion rate, tool-call
success rate, trajectory efficiency (steps/tokens per solved task)."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.schemas import RunResult, RunStatus


@dataclass
class TrajectoryMetrics:
    task_completion_rate: float
    tool_call_success_rate: float
    mean_steps_per_solved_task: float
    mean_tokens_per_solved_task: float
    n_runs: int

    def summary(self) -> dict:
        return {
            "task_completion_rate": self.task_completion_rate,
            "tool_call_success_rate": self.tool_call_success_rate,
            "mean_steps_per_solved_task": self.mean_steps_per_solved_task,
            "mean_tokens_per_solved_task": self.mean_tokens_per_solved_task,
            "n_runs": self.n_runs,
        }


def compute_trajectory_metrics(
    results: list[RunResult], tool_calls_by_run: dict[str, list[dict]]
) -> TrajectoryMetrics:
    n = len(results)
    completed = [r for r in results if r.status == RunStatus.COMPLETED]
    completion_rate = len(completed) / n if n else 0.0

    all_tool_calls = [tc for calls in tool_calls_by_run.values() for tc in calls]
    success_rate = sum(1 for tc in all_tool_calls if tc["success"]) / len(all_tool_calls) if all_tool_calls else 0.0

    steps_per_solved = [len(tool_calls_by_run.get(r.run_id, [])) for r in completed]
    tokens_per_solved = [r.total_tokens_in + r.total_tokens_out for r in completed]

    return TrajectoryMetrics(
        task_completion_rate=completion_rate,
        tool_call_success_rate=success_rate,
        mean_steps_per_solved_task=(sum(steps_per_solved) / len(steps_per_solved)) if steps_per_solved else 0.0,
        mean_tokens_per_solved_task=(sum(tokens_per_solved) / len(tokens_per_solved)) if tokens_per_solved else 0.0,
        n_runs=n,
    )
