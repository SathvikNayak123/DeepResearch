# Rerank ablation — results and decision

Read `docs/DESIGN.md` decision row 7 first. That row committed to shipping a
self-hosted cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) behind a
swappable `RerankBackend` interface, with a with/without-rerank ablation
before defaulting it on. This is that ablation.

## Method

**No LLM judge, no full agent loop.** This measures the rerank stage in
isolation against ground truth — a query, a candidate pool, and which
candidates are actually relevant, all known in advance.

- **Data**: 50 questions sampled (seed 42, no stratification — see
  "what this doesn't test" below) from `bdsaglam/musique`, `answerable`
  config, `validation` split (CC BY 4.0). Each question ships ~20 candidate
  paragraphs with a binary `is_supporting` gold label — 2-hop MuSiQue
  questions typically have exactly 2 supporting paragraphs among the ~20.
  Verified before running that supporting-paragraph positions are scattered
  through the list (not front-loaded), so "paragraph order as given" is a
  fair stand-in for "whatever order an imperfect, non-relevance-sorted
  retrieval step returns" — the thing rerank is meant to fix.
- **Raw baseline**: candidates in dataset-given order (no rerank).
- **Reranked**: `BAAI/bge-reranker-v2-m3` via `sentence-transformers`
  `CrossEncoder`, scoring every `(question, paragraph)` pair, sorted by
  score descending. This is the exact `CrossEncoderRerankBackend` class
  used in production (`src/deepresearch/rerank/bge.py`), not a stand-in.
- **Metrics** at k ∈ {3, 5}: hit_rate@k (≥1 relevant in top-k), recall@k
  (fraction of relevant docs captured in top-k), nDCG@k (binary relevance).
- **Latency**: wall-clock per question for the rerank call, model already
  loaded/warmed before timing starts (first call is excluded).

Full config and raw metric values: `results/rerank_ablation_20260702T120737Z.json`.

```json
{
  "dataset": "bdsaglam/musique",
  "dataset_config": "answerable",
  "dataset_split": "validation",
  "n_questions": 50,
  "seed": 42,
  "k_values": [3, 5],
  "rerank_model": "BAAI/bge-reranker-v2-m3"
}
```

## Results

| Metric | Raw (no rerank) | Reranked | Delta |
|---|---|---|---|
| hit_rate@3 | 0.34 | 0.96 | **+0.62** |
| recall@3 | 0.16 | 0.70 | **+0.54** |
| nDCG@3 | 0.152 | 0.730 | **+0.578** |
| hit_rate@5 | 0.48 | 0.98 | **+0.50** |
| recall@5 | 0.247 | 0.822 | **+0.575** |
| nDCG@5 | 0.197 | 0.790 | **+0.592** |

**Latency** (bge-reranker-v2-m3, CPU, batch of ~20 pairs per question, this
dev machine): mean 13.8s / p50 13.7s / p95 20.3s per rerank call.

## Decision

**Rerank stays ON by default** (`RunConfig.rerank_enabled = True`,
`rerank_backend = "bge"` — unchanged from the pre-ablation default).

The quality delta is not close: nearly 3x recall@3 and hit_rate@3 going from
0.34 to 0.96 means the un-reranked worker misses the actual answer-bearing
passage in top-3 roughly two-thirds of the time on this sample, while the
reranked worker misses it about once in 25 questions. Given the whole point
of the worker stage is grounding claims in the *right* source, this is a
clear case where the quality delta dominates.

**The latency cost is real and worth recording honestly, not averaging
away.** ~14s per sub-question on this CPU is a meaningful chunk of a
worker's wall-clock budget (`BudgetConfig.max_wall_clock_seconds` defaults
to 600s for the whole run, across parallel workers — one rerank call per
worker doesn't serialize against others, but it's not free). This matches
the risk docs/DESIGN.md row 7 already named: *"Self-hosted CPU reranker
latency becomes the dominant contributor to end-to-end p95 and no GPU is
affordable/available."* That hasn't flipped the decision here because the
quality delta is large enough to absorb it, but it's the first thing to
re-measure if p95 end-to-end latency becomes a problem in practice — at
which point the documented mitigation is to swap the *default* to the
already-implemented `CohereRerankBackend` (same interface, no code change
in `worker.py`) for the live path, keeping the self-hosted model for
CI/ablations where $ cost matters more than latency.

**What this doesn't test** (explicitly, so it isn't mistaken for more than
it is):
- No stratification by hop count — the 50-question sample is a plain random
  draw from the validation split, not balanced 2/3/4-hop like the eval
  design's FRAMES/MuSiQue subsets will be (that's Session 4 territory).
- Candidates here come pre-assembled by the dataset, not by an actual
  `search_backend.search()` call — the ablation isolates the reranker's
  ability to discriminate relevant from irrelevant text in a mixed pool,
  independent of how that pool was assembled. Whether Tavily's raw result
  order is *as bad as* MuSiQue's shuffled order is a separate, unmeasured
  question (docs/DESIGN.md row 5's own reversal-evidence column already
  flags Tavily extraction quality as something to spot-check).
- This is a retrieval-quality ablation only — it says nothing about whether
  better retrieval moves the needle on final citation accuracy or report
  quality, which needs the full FRAMES/MuSiQue + judge pipeline (Session 4).

## Reproducing

```bash
pip install -e ".[dev,eval]"
python scripts/rerank_ablation.py --n 50 --seed 42 --k 3 5
```

Writes a fresh timestamped JSON to `results/`. Swap `--model` to compare a
different cross-encoder, or point `DEEPRESEARCH_RERANK_BACKEND=cohere` at a
run of the full agent to compare against the hosted alternative end-to-end.

---

# Cache layer — cold/warm/mixed measurement

Read `docs/DESIGN.md` decision row 8 first. That row committed to Redis
caching search results (keyed on normalized query) and fetched pages (keyed
on canonical URL), with a one-flag bypass for eval runs that must be cold.
This is that measurement.

## Method

- **Real production classes under test**: `CachedSearchBackend` and
  `RedisCache` (`src/deepresearch/backends/cached.py`,
  `src/deepresearch/cache/redis_cache.py`) — unmodified, same code path the
  agent uses.
- **`FakeTavilyBackend`** stands in for the real Tavily API: this sandboxed
  session has no live Tavily key/credits (same constraint as the rerank
  ablation's environment). It simulates realistic network latency
  (`search`: 0.2–0.4s, `fetch`: 0.15–0.35s, uniform random) but makes no real
  HTTP calls. **This is the load-bearing caveat on every number below** —
  see "Failure honesty" for exactly what it does and doesn't tell us.
- **Questions**: 20 real MuSiQue questions (`bdsaglam/musique`, `answerable`,
  `validation`, seed 42 — same source as the rerank ablation), plus a
  disjoint 20-question sample (seed 1042) for the "fresh" half of the mixed
  pass.
- **Cost accounting**: `search_cost_usd = $0.008` (Tavily basic search, 1
  credit @ $0.008/credit), `fetch_cost_usd = $0.0016` (Tavily extract, 1
  credit per 5 URLs, amortized per single-URL fetch call) — same constants
  `RunConfig` uses for the real agent's `$-saved` accounting.
- **Redis backend**: the harness tries a real Redis at `REDIS_URL` first;
  none was reachable in this sandbox (no Docker daemon here — same
  constraint noted in earlier sessions), so it fell back to in-process
  `fakeredis`. The `RedisCache` class itself is identical either way — only
  the socket underneath differs.
- **Four passes**, sharing one cache instance except where noted:
  1. **cold** — the 20 questions against an empty cache.
  2. **warm** — the *same* 20 questions again, right after.
  3. **mixed** — 10 of the same questions + 10 fresh ones (simulates
     overlapping topics across separate runs, not a clean cold/warm split).
  4. **bypass** — the same 20 questions again, but with `CachedSearchBackend`
     removed entirely (`cache_enabled=false`'s effect), proving the one-flag
     bypass actually bypasses rather than just skipping writes.

Full config + raw numbers: `results/cache_measurement_20260702T122421Z.json`.

## Results

| Pass | Wall-clock | Hit rate | $ spent | $ saved |
|---|---|---|---|---|
| cold (empty cache) | 21.54s | 0.00 | $0.2560 | $0.0000 |
| warm (same 20 questions) | **0.01s** | 1.00 | $0.0000 | $0.2560 |
| mixed (10 repeat + 10 fresh) | 11.01s | 0.50 | $0.1280 | $0.1280 |
| bypass (`cache_enabled=false`) | 21.52s | 0.00 | $0.2560 | $0.0000 |

Config: 20 questions, 3 fetches/question (60 search calls + 60 fetch calls
total per pass, i.e. 20+60=80 cache-checkable events), seed 42.

## Reading this honestly

- **Warm vs. cold is close to the theoretical ceiling** (21.54s → 0.01s,
  ~2000x) because every single call in the warm pass is a hit — this is the
  *best possible* case (identical question set, no TTL expiry, no eviction),
  not a realistic steady-state number. Production hit rate will sit between
  the mixed pass's 0.50 and the warm pass's 1.00 depending on how much
  sub-question overlap actually occurs across runs.
- **Mixed is the more representative number**: 50% hit rate (by construction
  of this test, not measured from real traffic) still cuts spend in half and
  wall-clock by ~49% (21.54s → 11.01s) versus fully cold — caching pays for
  itself even at partial overlap.
- **`$` figures come from Tavily's published per-call pricing, not a real
  invoice.** No live Tavily calls were made — see "What this doesn't test."
- **Latency figures are dominated by the simulated network sleep, not real
  variance.** `asyncio.sleep(uniform(0.2, 0.4))` has none of a real API's
  tail latency, retries, or rate-limit backoff — treat the *relative* deltas
  (cold vs. warm vs. mixed) as informative, the *absolute* seconds as not.

## Failure honesty (per this session's brief)

- **Staleness is the real cost of the TTL choice, and this measurement
  doesn't surface it.** 24h (search) / 7d (fetch) TTLs mean a cached page
  can silently diverge from the live page for up to a week — e.g. a
  breaking-news source updated mid-week, a Wikipedia edit after a vandalism
  revert, a changelog page. docs/DESIGN.md's own risk table names this
  ("Measured staleness causes eval-answer drift") with the stated
  mitigation (shorter TTL on volatile domains) — that mitigation isn't
  implemented yet; today every URL gets the same 7-day TTL regardless of
  domain volatility. Worth revisiting once Session 4's eval harness can
  actually detect an eval-answer drift caused by a stale cache entry, rather
  than guessing which domains are "volatile."
- **All-fresh queries get zero benefit, by construction.** If every
  sub-question in a run is genuinely novel (no repeat questions, no shared
  URLs across sub-questions or across runs), the cache does nothing but add
  a Redis round-trip's worth of latency per call before the guaranteed miss.
  The "cold" pass above *is* this case — 21.54s either way (cached-and-empty
  vs. no-cache-at-all), confirming the cache doesn't hurt a first encounter,
  but it categorically can't help one either. Whether real DeepResearch
  traffic looks more like "cold" (research questions are inherently novel)
  or "mixed" (sub-questions across different top-level questions converge on
  the same well-known sources) is an open empirical question this harness
  can't answer — it needs real traffic, not simulated questions.
- **What this doesn't test**: no real Redis (fakeredis stood in — same
  interface, but no network hop, no memory pressure, no eviction under
  load); no real Tavily latency/rate-limiting/error modes; no TTL expiry
  observed in practice (all passes ran in seconds, nowhere near 24h/7d); no
  measurement of the Prometheus/Grafana hit-rate panels against live
  traffic (this sandbox has no Docker daemon to run `docker compose up` —
  the dashboard JSON is provisioned and shaped correctly but unverified
  against a running Grafana instance; verify by running `make up` and
  hitting `/research` a few times against real keys).

## Reproducing

```bash
pip install -e ".[dev,eval]"
python scripts/cache_measurement.py --n 20 --seed 42 --fetches 3
```

Point `REDIS_URL` at a real Redis to measure against it instead of
`fakeredis`. Set `TAVILY_API_KEY` and swap `FakeTavilyBackend` for the real
`TavilyBackend` in the script to get a live-latency measurement once keys
are available.

---

# Evals-as-a-system: run store, local corpus, benchmark harness, first baseline

Read `docs/DESIGN.md` §2 row 9 (run store), §5 (eval design), and the session
map's Session 4 row first. This session built all of it: the Postgres/SQLite
run store, a real per-question local-corpus backend, FRAMES + MuSiQue
benchmark loaders and scoring, judge economics, and the reliability job —
then ran them for real against this sandbox's constraints (below).

## What's real vs. simulated in this run — read this before the numbers

**No `ANTHROPIC_API_KEY` was available in this sandbox** (same constraint as
every prior session touching a paid external service). Every run below used
`eval.fake_llm.FakeLLMClient` — a stand-in that never reasons about the
question, sometimes echoes a random real snippet of the provided source text
back as its "answer," and (for judge calls) returns a random verdict at a
fixed probability. **Every accuracy/F1/reliability number in this document is
harness validation, not a measurement of DeepResearch's actual research
quality.** What it *does* prove, honestly:

- The full pipeline — plan -> parallel workers -> rerank -> reflection ->
  synthesis -> judge -> score -> persist — runs end-to-end with zero
  exceptions across 100+ real agent invocations, against real BM25
  retrieval over real benchmark corpora.
- The run store actually receives `runs`/`trajectories`/`tool_calls`/
  `eval_scores` rows with correct foreign keys, correct stage names, and
  non-placeholder (if zero, because the fake client reports zero) cost/token
  figures.
- The reliability job produces a genuine distribution (not a fabricated one)
  because the fake client's judge calls really do draw from a random
  distribution each call.

To get real baseline numbers: set `ANTHROPIC_API_KEY`, re-run the exact same
commands below (`make eval-smoke` / `eval-full` / `eval-reliability`) — the
harness auto-detects the key and switches to the real `LLMClient` with no
other change. Nothing in the harness is fake-client-specific except that one
`if` in `eval/run_eval.py:make_llm()`.

**Also this sandbox has no Docker daemon running and no `make` binary on
PATH** — commands below were run as the raw `python -m ...` invocations the
Makefile targets wrap (identical behavior; `make` just isn't installed
here). And no live Postgres — every run below used the SQLite dev-loop swap
(`sqlite+aiosqlite:///./deepresearch.db`), per docs/DESIGN.md decision row 9.

## Run store

Schema exactly matches docs/DESIGN.md §4 (`runs`, `trajectories`,
`tool_calls`, `eval_scores`, `ci_baselines`), plus one addition —
`judge_cache`, needed for this session's judge-cost task (see `models.py`
for the reasoning). `src/deepresearch/store/models.py` is the single source
of truth; `db/migrations/0001_init.sql` is generated from it
(`scripts/gen_migration.py`), not hand-maintained, so the two can't drift.
Portable by design: `postgresql+asyncpg://` in CI/deployed,
`sqlite+aiosqlite://` for local dev — same code, same schema, different
connection string.

Every `run_research()` call now writes a `runs` row at start and
`trajectories`/`tool_calls`/`finish_run` at the end automatically — this
isn't opt-in, and there's no code path that skips it (verified in
`tests/test_orchestrator_persistence.py`).

## Local corpus backend

Rewritten from a stub into a real BM25 lexical-retrieval backend
(`rank-bm25`), scoped to **one benchmark question's candidate pool per
instance** — MuSiQue ships its own paragraphs (gold + distractors) directly;
FRAMES only ships Wikipedia *links*, so its corpus is built by actually
fetching each linked article once via the MediaWiki action API (the REST
`page/plain` endpoint 403'd/404'd in testing; the classic
`action=query&prop=extracts` endpoint worked reliably) and caching the text
to `data/corpus/{frames,musique}/*.json`.

## Benchmark loaders

| Benchmark | Source | Revision pinned | Sampling |
|---|---|---|---|
| FRAMES | `google/frames-benchmark`, 824 questions | `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef` | seed 42, stratified by `reasoning_types` |
| MuSiQue | `bdsaglam/musique`, `answerable` config, `validation` split | `22873a405dd809893b22ada0b499299fb612d2df` | seed 42, stratified by hop count (2/3/4) |

Both revisions captured via `huggingface_hub.HfApi().dataset_info(...)` at
build time — docs/DESIGN.md's own risk table names silent benchmark drift as
a real risk; pinning is the stated mitigation.

## Judge economics

Default judge: `claude-haiku-4-5` (cheapest tier) — a deliberate exception to
"use the strongest model," made because grading doesn't need frontier
capability (docs/DESIGN.md §5.5). Verdicts cached in `judge_cache`, keyed on
`sha256(rubric_version + example + produced_answer)` — a second identical
run against unchanged content pays $0 in judge cost, verified in
`tests/test_store.py::test_judge_cache_is_idempotent_on_race`. Cost estimate
printed before every run (`eval.judge.estimate_judge_cost_usd`, from a
documented per-call token-count planning assumption, not a measured
average yet) and the real total (`judge.calls_made`, `judge.cache_hits`,
`judge.total_cost_usd`) is in every summary below.

## Baseline: eval-smoke (n=20, both benchmarks)

```
python -m eval.run_eval --mode smoke
```

Full config + raw scores: `results/eval_smoke_20260702T154642Z.json`. All 40
runs (20 FRAMES + 20 MuSiQue) completed with zero exceptions.

| Benchmark | n | Wall-clock | Task completion | Tool-call success | Mean tokens/solved | Metric(s) |
|---|---|---|---|---|---|---|
| FRAMES | 20 | 1703.8s | 1.00 | 1.00 | 1850 | accuracy 0.70, citation coverage 1.00, citation precision 0.70 |
| MuSiQue | 20 | 118.4s | 1.00 | 1.00 | 1324 | answer_f1 0.018, answer_contains_gold 0.05 |

Judge: 40 calls made (FRAMES only — accuracy + citation checks), 0 cache
hits (cold cache, first run), $0.00 actual cost (fake client). Estimated
before running: agent ~$3.00 + judge ~$0.03 per benchmark.

**Citation coverage pinned at 1.00 is a fake-client artifact, not a result**
— the fake worker/synthesis stages always cite `src_1`, so coverage is
mechanically 1.0 regardless of content. Re-measure with a real model before
citing this number for anything.

### A real, measured finding buried in this run: corpus shape drives rerank cost

FRAMES took **14.4x longer wall-clock** than MuSiQue for the same n=20, and
it isn't judge or LLM latency — it's the reranker. Querying the run store
directly:

| Benchmark | Mean rerank candidates/call | Mean rerank latency/call |
|---|---|---|
| FRAMES | 146.5 chunks | 84.7s |
| MuSiQue | 7.1 chunks | 5.5s |

FRAMES documents are full Wikipedia articles, chunked into ~800-char pieces
(`chunk_text`, `src/deepresearch/chunking.py`) — one article often yields
10-20+ chunks. MuSiQue documents are already short, pre-chunked paragraphs
— almost no further splitting happens. The reranker (self-hosted
`bge-reranker-v2-m3`, CPU) scores every chunk against the query, so cost
scales with total chunk count, not document count. This is a direct,
measured consequence of the rerank ablation's own finding (docs/DESIGN.md
row 7: "self-hosted CPU reranker latency becomes the dominant contributor to
end-to-end p95") — showing up exactly where predicted, on a corpus shape
the ablation didn't test. Worth a follow-up: cap chunks-per-source or
sub-sample large documents before reranking when the corpus is
full-document rather than pre-chunked.

## Baseline: eval-full, MuSiQue only (n=100)

```
python -m eval.run_eval --benchmark musique --n 100 --seed 42
```

Full output: `results/eval_custom_20260702T160417Z.json`. All 100 runs
completed with zero exceptions.

| n | Wall-clock | Task completion | Tool-call success | Mean tokens/solved | answer_f1 | answer_contains_gold |
|---|---|---|---|---|---|---|
| 100 | 615.4s (~5.9x the n=20 wall-clock, roughly linear) | 1.00 | 1.00 | 1380 | 0.022 | 0.04 |

Scaling from n=20 (118.4s) to n=100 (615.4s) is close to linear (5.2x time
for 5x questions) — expected, since MuSiQue's rerank cost per question is
small and roughly constant (§ above), unlike FRAMES where it isn't.

## eval-full, FRAMES — not run this session (honest, not an oversight)

Extrapolating from the smoke run's measured 1703.8s for 20 questions, a
100-question FRAMES-full run would take on the order of **2-2.5 hours**
(rerank cost scales with total chunks fetched, not linearly guaranteed, but
roughly proportional to n). That's outside this session's time budget on top
of everything else built here. The code path is identical to the n=20 run
already proven above (`--mode full` just changes `n`) — this is exactly the
kind of item this session's own brief flagged as splittable into a follow-up
sitting. Recommended before running it for real: address the chunking/rerank
cost finding above first, or the 100-question run will be dominated by the
same inefficiency at 5x the scale.

## Reliability job (20 questions x 3 repeats)

```
python -m eval.run_eval --reliability --n 20 --repeats 3
```

Full output: `results/eval_reliability_20260702T155323Z.json`.

| Metric | Value |
|---|---|
| Per-repeat accuracy | [0.05, 0.10, 0.05] |
| Mean accuracy | 0.067 |
| Stdev accuracy | 0.024 |
| All-consistent (pass^k) rate | 0.80 |

Reported as a distribution, not a point estimate, per CLAUDE.md ("an
accuracy figure without [reliability] is incomplete and should not be cited
on its own"). 16/20 questions got the same verdict all 3 repeats; 4/20
flipped at least once — with a real model this would be the signal that
those 4 questions are borderline/ambiguous for the agent, not benchmark
noise. With the fake client it's just confirmation the variance-reporting
mechanics work (the fake judge draws an independent random verdict every
call, so *some* disagreement is expected and not itself meaningful here).

## DeepResearch Bench

Per docs/DESIGN.md decision row 11, judge-cost analysis did **not** approve
this for nightly (or weekly-full). This session shipped only the gated
manual path: `make eval-drb` prints the documented cost estimate
($15-35/weekly-10q, $120-330/full-100q) and requires `--confirm` to proceed
past it; past that gate it's an explicit, honest `NotImplementedError` — the
actual RACE/FACT judge pipeline (`Ayanami0730/deep_research_bench`) wasn't
in scope for this session (see `eval/benchmarks/deepresearch_bench.py`'s
module docstring for the reasoning). This is the one piece of task 3
deliberately left as a stub rather than faked into looking done.

## Reproducing

```bash
pip install -e ".[dev,eval]"
python scripts/migrate.py                          # apply schema (or rely on lazy init_schema)
python -m eval.run_eval --mode smoke                # ~20q both benchmarks
python -m eval.run_eval --mode full                 # ~100q both benchmarks (FRAMES: budget hours, see above)
python -m eval.run_eval --reliability --n 20 --repeats 3
python -m eval.benchmarks.deepresearch_bench --mode weekly --confirm   # honest NotImplementedError
```

Set `ANTHROPIC_API_KEY` for real model scores; set `DATABASE_URL` to point
at a real Postgres instead of the default SQLite file.

# CI + the architecture ablation (2026-07-02)

Session brief: GitHub Actions PR gate + nightly workflow, the plan-first-vs-
ReAct / worker-pool-size ablation, and a deliberately-broken PR proving the
gate actually catches a regression. Repo: `github.com/SathvikNayak123/DeepResearch`
(this session's first `git init` — everything before this point in this file
was built and measured without a git repo backing it; `git_sha` on all of
those runs reads `no-git`, which is an accurate historical record, not a bug).

## Architecture ablation

Full method, table, and findings are in `docs/DESIGN.md` §10 (dated
addendum) — not duplicated here to avoid the two docs drifting apart. Short
version: `results/architecture_ablation_20260702T164109Z.json`, MuSiQue
smoke subset (n=20), three variants (`plan_first_pool4` default,
`plan_first_pool1` worker-pool sweep, `react` planning-style alternative).
Headline, real (not `FakeLLMClient`-artifact) finding: ReAct costs ~1.9x
plan-first's tokens and ~1.8x its wall-clock per question, mechanically —
one extra LLM round-trip per query decided, versus plan-first's one round-
trip total. Neither decision-table row (1 or 2) is reversed. Same sandbox
caveat as everywhere else in this file: no `ANTHROPIC_API_KEY`, so the
accuracy/F1 columns are not trustworthy signal and are intentionally left
out of the table in DESIGN.md — only steps/tokens/latency are real
measurements here.

## CI regression gate

`.github/workflows/pr-smoke.yml`: on every PR into `main`, runs
`eval-smoke` (FRAMES 20q + MuSiQue 20q) against `LocalCorpusBackend` with no
`ANTHROPIC_API_KEY`/`TAVILY_API_KEY` secret configured — `make_llm()`'s
existing `FakeLLMClient` fallback (built in the evals-as-a-system session)
means this "just works" with no CI-specific code path, matching this
session's brief ("no external APIs in CI; judge calls cached/stubbed per
design"). `scripts/ci_gate.py` then compares the run's metrics against
`results/ci_baseline.json` and fails the job (naming the offending metric)
if:
- `frames.accuracy` or `musique.answer_f1` drops >3 points absolute
- `frames.citation_precision` drops >3 points absolute
- `frames.cost_per_query_usd` or `musique.cost_per_query_usd` rises >25% relative
- `frames.task_completion_rate` or `musique.task_completion_rate` drops >3
  points absolute (added mid-session — see the finding right below)

**A real gap this session's own sanity-checking surfaced, before pushing
the deliberately-broken demo PR**: none of the first three metrics above
can actually be moved by a real agent-side regression when CI has no
`ANTHROPIC_API_KEY` (i.e. every PR-smoke run, by this session's own design).
`frames.accuracy`/`citation_precision` are judge-scored, and
`FakeLLMClient`'s judge branches (`eval/fake_llm.py`) return
`self._rng.random() < 0.7` / `< 0.8` — a verdict with **zero dependence on
the run's actual content**, so no code change can move their aggregate away
from ~70%/~80% ± sampling noise. `musique.answer_f1`'s own baseline
(0.021) sits so close to its floor that even total answer collapse (every
report empty) only drops it by ~2 points — under the 3-point tolerance.
Verified this the hard way: the originally-planned deliberately-broken
change (`candidate_pool_size` 6→0, cutting off all retrieved content)
measured a negligible, noise-level shift in `answer_f1`/`answer_contains_gold`
locally, not the clear regression expected. Added `task_completion_rate`
(`runs.status == "completed"` fraction — already a designed agentic metric,
docs/DESIGN.md §5.2) as a fourth gated metric specifically because it has
none of these blind spots: it doesn't route through the judge at all, and
it isn't floor-bound. The deliberately-broken PR below targets this metric.

These are this session's literal numbers, not CLAUDE.md's placeholder
5-point/30%-latency thresholds — CLAUDE.md's own text calls those
"placeholders until the first real baseline lands," so tightening them here
(once a first real baseline did land, see below) is the intended next step,
not a contradiction. Reconciling the two documents' numbers is a flagged
follow-up, not done this session.

**Baseline persistence**: CI runners are ephemeral and this project has no
externally-hosted Postgres reachable from GitHub Actions (docs/DESIGN.md
decision row 9's "Postgres in CI" assumes a reachable instance this sandbox
doesn't have), so `results/ci_baseline.json` — a checked-in JSON snapshot,
same config-next-to-result shape as the `ci_baselines` table — is what
PR-smoke reads and what nightly refreshes on green
(`scripts/dump_ci_baseline.py`), rather than a live `ci_baselines` query.
`db.get_latest_ci_baseline` still exists and is unit-tested
(`tests/test_store.py`) for a future deployment with a persistent CI
database to swap back to.

**First real baseline**: seeded from this project's own already-verified
evals-as-a-system-session data (`frames` n=20, `musique` n=20 smoke + n=100
full = 120 completed runs, all `status=completed`), not re-run this session
— same `FakeLLMClient` constraint applies either way, so re-running would
have cost ~30-40 minutes (FRAMES' measured rerank latency) for the same
kind of number. `results/ci_baseline.json`:

| Metric | Value |
|---|---|
| `frames.accuracy` | 0.700 |
| `frames.citation_precision` | 0.700 |
| `frames.cost_per_query_usd` | $0.0000 (FakeLLMClient — real cost needs a live key) |
| `frames.task_completion_rate` | 1.000 |
| `musique.answer_f1` | 0.021 |
| `musique.cost_per_query_usd` | $0.0000 |
| `musique.task_completion_rate` | 1.000 |

`.github/workflows/nightly.yml`: `eval-full` + the reliability job, uploads
`results/*.json` as an artifact every run (`if: always()`), and only commits
a refreshed `results/ci_baseline.json` back to `main` if every run recorded
in that invocation's database finished `status=completed`
(`scripts/dump_ci_baseline.py`'s green check) — matches CLAUDE.md's "no
auto-gate" nightly policy (the job itself never fails red), while still
refusing to quietly lower the bar from a crashed/budget-exceeded night.

## Proving the gate works: bootstrap, clean, and deliberately-broken PRs

Real GitHub repo, real push access (verified with a disposable probe branch/
push/delete before doing anything else this session), but no `gh` CLI and no
extracted token in this sandbox — creating the PR objects themselves needs a
human click on the `pull/new/<branch>` URL `git push` already prints, since
PR creation and PR-comment-posting both go through GitHub's API rather than
git's own transport. The workflow: three branches pushed by this session —

1. `bootstrap-ci-and-ablation` — everything in this session (plus the
   accumulated prior-session code this repo's `main` didn't have yet, since
   this was this project's first `git init`). PR gate on this one should
   **bootstrap** (no baseline existed on `main` before it) — see
   `scripts/ci_gate.py`'s bootstrap path above.
2. `demo/ci-gate-clean` — a no-op-equivalent change, should go **green**
   with a before/after/delta comment table showing no regression.
3. `demo/ci-gate-broken` — one deliberate regression:
   `BudgetConfig.max_total_tokens` dropped from `200_000` to `10`
   (`src/deepresearch/config.py`) — a very plausible real mistake (a typo,
   or a units mix-up between "tokens" and "thousands of tokens"). The
   planner's own first call already spends more than 10 tokens, so
   `budget.check()` raises `BudgetExceeded` before a single worker runs,
   for every question, deterministically, regardless of LLM backend —
   `runs.status` becomes `budget_exceeded` instead of `completed` for the
   whole benchmark, crashing `task_completion_rate` from 1.0 to 0.0. This
   replaces an earlier plan (disabling rerank / cutting `candidate_pool_size`
   to 0) that, on local sanity-checking before push, turned out to barely
   move any FakeLLMClient-scored metric at all — see the finding above.

[Outcomes of PRs 2 and 3 filled in below once opened — see the bottom of
this section for the actual red/green results and the PR links.]
