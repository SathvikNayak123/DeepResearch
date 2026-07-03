"""Apply the run-store schema to config.database_url (or $DATABASE_URL).

Idempotent (CREATE TABLE IF NOT EXISTS under the hood) — safe to run against
an already-migrated database. For the canonical Postgres DDL as plain SQL
(e.g. to hand to a DBA or a migration tool), see db/migrations/0001_init.sql
(scripts/gen_migration.py regenerates it from store/models.py).

Usage:
    python scripts/migrate.py [DATABASE_URL]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.store import db  # noqa: E402


async def main() -> None:
    database_url = sys.argv[1] if len(sys.argv) > 1 else RunConfig().database_url
    print(f"Applying schema to {database_url} ...")
    await db.init_schema(database_url)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
