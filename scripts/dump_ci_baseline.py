"""Nightly baseline refresh — docs/DESIGN.md §6 ("nightly.yml ... updates the
baseline only on green"). "Green" here means every run recorded across all
given --database-url values finished with status=completed; CLAUDE.md's
nightly policy is "no auto-gate" for blocking anything, but a nightly
baseline should never be refreshed from a run that itself didn't finish
cleanly (budget_exceeded or failed runs would silently lower the bar for
every future PR).

--database-url may be passed more than once. Nightly runs FRAMES-full and
MuSiQue-full as separate parallel GitHub Actions jobs (each on its own
runner, own SQLite file) so FRAMES' rerank-dominated wall-clock doesn't
force MuSiQue and the reliability job to share its timeout budget — this
script merges the resulting metrics dicts by key (frames.* / musique.* never
collide) rather than needing one shared database.

Usage:
    python scripts/dump_ci_baseline.py --database-url sqlite+aiosqlite:///./frames_full.db --database-url sqlite+aiosqlite:///./musique_full.db
Exit code 1 (baseline NOT updated) if any recorded run in any given database
isn't status=completed — the nightly workflow commits results/ci_baseline.json
only on exit 0.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from deepresearch.store import db  # noqa: E402
from deepresearch.store.models import runs  # noqa: E402

from eval.ci_baseline import compute_current_metrics, write_baseline  # noqa: E402

DEFAULT_BASELINE_PATH = Path(__file__).parent.parent / "results" / "ci_baseline.json"


async def _all_runs_completed(database_urls: list[str]) -> tuple[bool, dict[str, int]]:
    """Combines run-status counts across every given database. A database
    whose schema was never created (an upstream job crashed before writing
    anything, so its artifact — and the file this URL points at — never
    existed) contributes zero rows, not a crash here — correctly treated as
    "not all green" below rather than taking down the whole refresh with an
    unrelated OperationalError."""
    counts: dict[str, int] = {}
    for database_url in database_urls:
        engine = db.get_engine(database_url)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(select(runs.c.status))
                statuses = [row[0] for row in result.fetchall()]
        except OperationalError:
            print(f"  (no `runs` table at {database_url} — treating as zero runs)")
            statuses = []
        for status in statuses:
            counts[status] = counts.get(status, 0) + 1
    all_green = bool(counts) and set(counts) <= {"completed"}
    return all_green, counts


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", required=True, action="append", help="repeatable")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()

    all_green, status_counts = await _all_runs_completed(args.database_url)
    print(f"Run status counts (across {len(args.database_url)} database(s)): {status_counts}")
    if not all_green:
        print(
            "Not all runs completed cleanly - baseline NOT updated. Per CLAUDE.md's "
            "nightly policy this is flagged for manual review, not auto-gated, so the "
            "workflow itself doesn't fail the job — it just skips the baseline commit."
        )
        return 1

    metrics: dict[str, float] = {}
    for database_url in args.database_url:
        metrics.update(await compute_current_metrics(database_url))
    baseline = write_baseline(args.baseline, metrics)
    print(f"Baseline updated at {args.baseline}:\n{baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
