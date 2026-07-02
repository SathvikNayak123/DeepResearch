"""PR-smoke CI regression gate — docs/DESIGN.md §6, this session's brief.

Compares the metrics from a just-finished eval-smoke run (in --database-url)
against the checked-in baseline (--baseline, default results/ci_baseline.json)
and fails (exit 1) if any tracked metric regresses beyond tolerance — see
eval/ci_baseline.py for the tolerances and the comparison/rendering logic
itself (kept there, not here, so it's importable and unit-tested without
shelling out — tests/test_ci_baseline.py).

Always writes a before/after/delta markdown table to --out, whether it
passes, fails, or bootstraps (see eval/ci_baseline.py's module docstring for
why "bootstrap" is a legitimate first-run outcome, not a bug).

Usage:
    python scripts/ci_gate.py --database-url sqlite+aiosqlite:///./ci_run.db
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepresearch.config import current_git_sha  # noqa: E402

from eval.ci_baseline import compute_current_metrics, fmt_metric, load_baseline, render_gate_table, write_baseline  # noqa: E402

DEFAULT_BASELINE_PATH = Path(__file__).parent.parent / "results" / "ci_baseline.json"
DEFAULT_OUT_PATH = Path(__file__).parent.parent / "results" / "ci_gate_comment.md"


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    current_metrics = await compute_current_metrics(args.database_url)
    baseline = load_baseline(args.baseline)
    sha = current_git_sha()[:8]

    if baseline is None:
        write_baseline(args.baseline, current_metrics)
        comment = (
            f"## Eval gate (git `{sha}`)\n\n"
            f"No baseline found at `{args.baseline}` — recording this run's metrics as the "
            f"bootstrap baseline. Nothing to gate against yet; this is expected on the first "
            f"PR-smoke run in a new repo, not a bypassed check.\n\n"
            f"| Metric | Value |\n|---|---|\n"
            + "\n".join(f"| `{k}` | {fmt_metric(k, v)} |" for k, v in sorted(current_metrics.items()))
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(comment, encoding="utf-8")
        print(comment)
        return 0

    table, failures = render_gate_table(baseline["metrics"], current_metrics)
    baseline_sha = baseline.get("git_sha", "unknown")
    result_line = f"**Result: FAIL** — {'; '.join(failures)}" if failures else "**Result: PASS**"
    comment = f"## Eval gate: `{sha}` vs baseline `{baseline_sha}`\n\n{table}\n\n{result_line}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(comment, encoding="utf-8")
    print(comment)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
