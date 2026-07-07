# DeepResearch

Read `docs/DESIGN.md` and `CLAUDE.md` first.

Planner -> parallel sub-question workers -> rerank -> reflection ->
synthesis, with Langfuse/OTel tracing, Redis-backed search/fetch caching,
hard budget enforcement, a Postgres/SQLite run store, and a FRAMES +
MuSiQue benchmark harness with a reliability job — all wired end-to-end.

Rerank (`BAAI/bge-reranker-v2-m3` by default, `Cohere` optional) and cache
(search results + fetched pages, Redis) are both on by default — see
`docs/RESULTS.md` for the ablation/measurement that earned each default
(quality delta, latency/cost deltas, full configs, `results/*.json`).

To force a cold, cache-bypassed run (e.g. for an eval run that must not see
stale cached content): set `DEEPRESEARCH_CACHE_ENABLED=false`, or pass
`{"config": {"cache_enabled": false}}` in a `POST /research` body.

## Setup

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
# generate Langfuse secrets: openssl rand -hex 32 (x3, for SALT/NEXTAUTH_SECRET/ENCRYPTION_KEY)
pip install -e ".[dev,eval]"
```

## Run

One command, one research question, one cited report:

```bash
python -m deepresearch.cli "Which came first, the Eiffel Tower or the Statue of Liberty?"
```

> **This live path needs real keys.** The CLI (and `POST /research`) call the real
> Anthropic + Tavily APIs — there is **no fake-client fallback on the live path**, so a
> keyless clean clone will fail here. For an offline, no-cost end-to-end run (frozen
> corpus + `FakeLLMClient`), use the eval harness below (`python -m eval.run_eval
> --mode smoke`) instead — that path *does* auto-fall-back with a loud banner.

Or via the API:

```bash
make up
curl -X POST localhost:8000/research -H "Content-Type: application/json" \
  -d '{"question": "Which came first, the Eiffel Tower or the Statue of Liberty?"}'
```

Langfuse UI: http://localhost:3000 (create an account, then set
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env` from a new project's API
keys, restart `app`). Each run's trace shares its `run_id` as the OTel trace
ID — search Langfuse for it to see the full connected trace.

## Test

```bash
make test         # unit tests, incl. budget-ceiling enforcement with a tiny budget
make demo          # 3 hand-picked live questions end-to-end, trajectories written to trajectories/
python scripts/rerank_ablation.py --n 50    # rerank ablation vs MuSiQue gold docs, see docs/RESULTS.md
python scripts/cache_measurement.py --n 20  # cold/warm/mixed/bypass cache measurement, see docs/RESULTS.md
```

## Run store + eval harness

```bash
python scripts/migrate.py           # apply the schema to $DATABASE_URL (or the SQLite default)
make eval-smoke                     # ~20q FRAMES + ~20q MuSiQue, local corpus, writes eval_scores
make eval-full                      # ~100q each — FRAMES-full is slow, see docs/RESULTS.md
make eval-reliability                # 20q x 3 repeats -> variance + all-consistent rate
make eval-drb                       # gated manual-only DeepResearch Bench stub, prints cost first
```

> **Local runs and the reranker.** Rerank is on by default, which pulls the ~1 GB
> `bge-reranker-v2-m3` cross-encoder on first use. On some
> `torch`/`sentence-transformers` combinations it also raises
> `NotImplementedError: Cannot copy out of meta tensor` on first load (a version
> incompatibility, not a logic bug). CI/nightly and the committed real baseline all
> run with `DEEPRESEARCH_RERANK_ENABLED=false`, which is a no-op for *selection*
> (`candidate_pool_size == rerank_top_k == 6`, so the same candidates are kept) and
> sidesteps both the download and the crash. Set it for local eval runs too unless you
> specifically want to exercise the reranker.

Every `run_research()` call — live or eval — writes a `runs` row (plus
`trajectories`/`tool_calls`) to `DATABASE_URL` automatically; defaults to a
local SQLite file (`sqlite+aiosqlite:///./deepresearch.db`, docs/DESIGN.md's
own documented dev-loop swap) if unset, or point it at
`postgresql+asyncpg://...` for the docker-compose `postgres` service. FRAMES
and MuSiQue both run against `LocalCorpusBackend` (real BM25 retrieval, no
network calls except FRAMES' one-time Wikipedia ingestion) so results are
reproducible and CI-safe.

**No `ANTHROPIC_API_KEY` set?** The harness auto-falls-back to a
`FakeLLMClient` and prints a loud banner — useful for verifying the
mechanics (it's exactly how this repo's own first baseline in
`docs/RESULTS.md` was produced), but treat any resulting scores as harness
validation, not real model performance. Set the key to get real numbers.

`make` isn't required — every target is a thin wrapper around a
`python -m ...` / `python scripts/...` invocation shown in the Makefile;
run those directly if `make` isn't installed.

## Observability

`GET /metrics` (Prometheus format) exposes `deepresearch_cache_hits_total` /
`deepresearch_cache_misses_total`, labeled by `cache_type` (`search`/`fetch`).
`make up` starts Prometheus (scrapes `app:8000/metrics` every 15s) and
Grafana (http://localhost:3001, anonymous viewer access, dashboard
"DeepResearch — Cache Hit Rate" provisioned automatically) alongside
Langfuse. Not verified against live traffic in dev (no Docker daemon in the
sandbox this was built in) — the panels are shaped and provisioned, but
confirm hit rates render by hitting `/research` a few times with real keys
and checking Grafana.

## Verifying budget enforcement

`make test` includes `tests/test_budget.py`, which sets `max_total_tokens`,
`max_usd`, and `max_wall_clock_seconds` to tiny values and asserts
`BudgetExceeded` is raised. To see it hit inside a real run, set
`DEEPRESEARCH_` budget overrides very low in the `config.budget` field of a
`POST /research` request body.
