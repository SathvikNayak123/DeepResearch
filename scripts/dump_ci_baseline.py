"""Nightly baseline refresh — docs/DESIGN.md §6 ("nightly.yml ... updates the
baseline only on green"). "Green" here means every run recorded in
--database-url finished with status=completed; CLAUDE.md's nightly policy is
"no auto-gate" for blocking anything, but a nightly baseline should never be
refreshed from a run that itself didn't finish cleanly (budget_exceeded or
failed runs would silently lower the bar for every future PR).

Usage:
    python scripts/dump_ci_baseline.py --database-url sqlite+aiosqlite:///./nightly_run.db
Exit code 1 (baseline NOT updated) if any recorded run isn't status=completed
— the nightly workflow commits results/ci_baseline.json only on exit 0.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select  # noqa: E402

from deepresearch.store import db  # noqa: E402
from deepresearch.store.models import runs  # noqa: E402

from eval.ci_baseline import compute_current_metrics, write_baseline  # noqa: E402

DEFAULT_BASELINE_PATH = Path(__file__).parent.parent / "results" / "ci_baseline.json"


async def _all_runs_completed(database_url: str) -> tuple[bool, dict[str, int]]:
    engine = db.get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(runs.c.status))
        statuses = [row[0] for row in result.fetchall()]
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return bool(statuses) and all(status == "completed" for status in statuses), counts


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()

    all_green, status_counts = await _all_runs_completed(args.database_url)
    print(f"Run status counts: {status_counts}")
    if not all_green:
        print(
            "Not all runs completed cleanly - baseline NOT updated. Per CLAUDE.md's "
            "nightly policy this is flagged for manual review, not auto-gated, so the "
            "workflow itself doesn't fail the job — it just skips the baseline commit."
        )
        return 1

    metrics = await compute_current_metrics(args.database_url)
    baseline = write_baseline(args.baseline, metrics)
    print(f"Baseline updated at {args.baseline}:\n{baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
