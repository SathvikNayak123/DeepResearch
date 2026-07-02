"""Run-store schema — docs/DESIGN.md §4, verbatim, plus one addition.

This is the single source of truth for the schema: db/migrations/0001_init.sql
is *generated* from this file (scripts/gen_migration.py), not maintained by
hand, so the two can't drift.

Portable across dialects on purpose (docs/DESIGN.md decision row 9): Postgres
in CI/deployed, SQLite for the local dev loop — same schema, different
connection string. JSONB/UUID use Postgres-native types with a SQLite variant
so `postgresql+asyncpg://...` and `sqlite+aiosqlite://...` both work against
this same MetaData.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    func,
)
from sqlalchemy.dialects import postgresql

metadata = MetaData()


def _uuid_type():
    return postgresql.UUID(as_uuid=False).with_variant(String(36), "sqlite")


def _json_type():
    return postgresql.JSONB().with_variant(JSON(), "sqlite")


def _bigint_pk_type():
    # SQLite only treats a literal INTEGER PK as the autoincrementing rowid
    # alias — BIGINT (what BigInteger compiles to there) silently isn't one,
    # so inserts NOT-NULL-fail with no id generated. Integer with a Postgres
    # variant gives each dialect the type it actually needs.
    return Integer().with_variant(BigInteger(), "postgresql")


runs = Table(
    "runs",
    metadata,
    Column("run_id", _uuid_type(), primary_key=True),  # also the OTel trace_id (32 hex)
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("benchmark_name", String, nullable=True),  # NULL for live/prod queries
    Column("config", _json_type(), nullable=False),  # config-next-to-result
    Column("git_sha", String, nullable=False),
    Column("status", String, nullable=False),  # running | completed | failed | budget_exceeded
    Column("total_cost_usd", Numeric(10, 4)),
    Column("total_latency_ms", Integer),
)

trajectories = Table(
    "trajectories",
    metadata,
    Column("id", _bigint_pk_type(), primary_key=True, autoincrement=True),
    Column("run_id", _uuid_type(), ForeignKey("runs.run_id"), nullable=False),
    Column("span_id", String, nullable=False, unique=True),  # OTel span id
    Column("parent_span_id", String),
    Column("stage", String, nullable=False),  # plan | worker | reflection | synthesis
    Column("name", String, nullable=False),
    Column("input", _json_type()),
    Column("output", _json_type()),
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("cost_usd", Numeric(10, 6)),
    Column("latency_ms", Integer),
    Column("started_at", DateTime(timezone=True)),
    Column("ended_at", DateTime(timezone=True)),
)

tool_calls = Table(
    "tool_calls",
    metadata,
    Column("id", _bigint_pk_type(), primary_key=True, autoincrement=True),
    Column("run_id", _uuid_type(), ForeignKey("runs.run_id"), nullable=False),
    Column("span_id", String, ForeignKey("trajectories.span_id"), nullable=False),
    Column("tool_name", String, nullable=False),  # search | fetch | rerank
    Column("args", _json_type()),
    Column("result_summary", _json_type()),
    Column("success", Boolean, nullable=False),
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("latency_ms", Integer),
)

eval_scores = Table(
    "eval_scores",
    metadata,
    Column("id", _bigint_pk_type(), primary_key=True, autoincrement=True),
    Column("run_id", _uuid_type(), ForeignKey("runs.run_id"), nullable=False),
    Column("benchmark_name", String, nullable=False),  # frames | musique | deepresearch_bench | trajectory | reliability
    Column("question_id", String),
    Column("metric_name", String, nullable=False),  # accuracy | answer_f1 | support_f1 | citation_precision | ...
    Column("value", Numeric),
    Column("judge_model", String),
    Column("rubric_version", String),
    Column("raw_judge_output", _json_type()),
)

ci_baselines = Table(
    "ci_baselines",
    metadata,
    Column("id", _bigint_pk_type(), primary_key=True, autoincrement=True),
    Column("benchmark_name", String, nullable=False),
    Column("metric_name", String, nullable=False),
    Column("baseline_value", Numeric, nullable=False),
    Column("config", _json_type(), nullable=False),  # config-next-to-result
    Column("git_sha", String, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

# Not in docs/DESIGN.md's original 5-table schema. This session's judge-cost
# task (docs/DESIGN.md §5.5: "Judge verdicts are cached by
# sha256(claim_text + source_id)") needs somewhere durable to live — this is
# that somewhere, sized deliberately small (one row per unique judged
# example+answer pair, keyed by the hash, judge model + rubric versioned
# same as eval_scores per the same design-doc paragraph).
judge_cache = Table(
    "judge_cache",
    metadata,
    Column("cache_key", String, primary_key=True),  # sha256(example + produced answer)
    Column("verdict", _json_type(), nullable=False),
    Column("judge_model", String, nullable=False),
    Column("rubric_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
