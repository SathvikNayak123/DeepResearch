.PHONY: up down test demo migrate eval-smoke eval-full eval-reliability eval-drb ci-gate dump-baseline logs

up:
	docker compose up --build -d

down:
	docker compose down

test:
	pytest -q

# 3 hand-picked live questions through the real Tavily backend — a sanity
# check of the live agent path, distinct from the benchmark harness below.
demo:
	python scripts/eval_smoke.py

migrate:
	python scripts/migrate.py

# ~20 questions per benchmark (FRAMES + MuSiQue), local corpus, writes
# scored eval_scores rows to the run store (Postgres/SQLite per DATABASE_URL).
eval-smoke:
	python -m eval.run_eval --mode smoke

# ~100 questions per benchmark, same pipeline as eval-smoke.
eval-full:
	python -m eval.run_eval --mode full

# 20 questions x 3 repeats -> variance + all-consistent (pass^k) rate.
# Never cite an accuracy number without this alongside it (CLAUDE.md).
eval-reliability:
	python -m eval.run_eval --reliability --n 20 --repeats 3

# Manual-only, gated: prints the documented cost estimate and requires
# --confirm. docs/DESIGN.md decision row 11 — not nightly-affordable.
eval-drb:
	python -m eval.benchmarks.deepresearch_bench --mode weekly

# Compares a just-finished eval-smoke run against results/ci_baseline.json;
# what .github/workflows/pr-smoke.yml invokes after eval-smoke.
ci-gate:
	python scripts/ci_gate.py --database-url "$${DATABASE_URL:-sqlite+aiosqlite:///./ci_run.db}"

# Regenerates results/ci_baseline.json from a completed eval-full + reliability
# run; what .github/workflows/nightly.yml invokes on green.
dump-baseline:
	python scripts/dump_ci_baseline.py --database-url "$${DATABASE_URL:-sqlite+aiosqlite:///./deepresearch.db}"

logs:
	docker compose logs -f app
