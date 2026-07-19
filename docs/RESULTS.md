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

## CI network flakiness: two incidents, two fixes

Getting the three demo PRs (below) to a trustworthy CI signal took two rounds
of "the code is right, the runner's network isn't":

1. **Corpus fetch.** The first 3 runs (all 3 branches) failed identically in
   ~2 min at the corpus-loading step. Local repro (fresh `HF_HOME`, deleted
   `data/corpus/`) succeeded fully, isolating it to something GitHub-Actions-
   runner-specific — most likely Wikipedia's edge WAF blocking GitHub's
   shared runner IP ranges (this same WAF's UA-sensitivity was already
   documented earlier in this project). Fix: froze the actual FRAMES (20
   files) and MuSiQue (100 files) corpus JSON used by `--mode smoke`/`--mode
   full` and committed them to git (`.gitignore` narrowed from a blanket
   `data/corpus/` exclusion to a pattern that keeps exactly those files) —
   CI no longer touches Wikipedia at all.
2. **Reranker model download.** With the corpus fix in, runs #4/#5
   (`bootstrap-ci-and-ablation`, `demo/ci-gate-clean`) hung 30+ min at
   `Generating test split` with no further progress and were manually
   cancelled; run #6 (`demo/ci-gate-broken`) finished in 1m46s because its
   deliberate regression (`BudgetConfig.max_total_tokens=10`) raises
   `BudgetExceeded` before the pipeline ever reaches the reranker, so it
   never hit the hang. Root cause: `BAAI/bge-reranker-v2-m3` is a ~1GB
   cross-encoder pulled from the HF model hub on first use per
   `src/deepresearch/rerank/bge.py`'s `_load()` — slow/stalled from GitHub's
   runner IPs, same class of issue as (1) but hitting the model hub instead
   of Wikipedia. Fix: `DEEPRESEARCH_RERANK_ENABLED=false` set as step-level
   `env` in both `pr-smoke.yml` and `nightly.yml`. This is a genuine no-op
   for every metric in scope here — `config.py`'s `candidate_pool_size` and
   `rerank_top_k` are both `6`, so `worker.py` selects the identical
   candidate set whether or not reranking runs; disabling it only removes
   the download, it does not change what CI measures.

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

**Actual outcomes** (after the rerank-download fix above; see that section
for why the first attempt at each of these needed a re-run):

| PR | Branch | Run | Conclusion | Comment |
|---|---|---|---|---|
| [#1](https://github.com/SathvikNayak123/DeepResearch/pull/1) | `bootstrap-ci-and-ablation` | [run 7](https://github.com/SathvikNayak123/DeepResearch/actions/runs/28644693703) | **success** | [PASS, no baseline drift](https://github.com/SathvikNayak123/DeepResearch/pull/1#issuecomment-4873670172) |
| [#2](https://github.com/SathvikNayak123/DeepResearch/pull/2) | `demo/ci-gate-clean` | [run 8](https://github.com/SathvikNayak123/DeepResearch/actions/runs/28644702439) | **success** | [PASS — all 7 gated metrics `OK`, `musique.answer_f1` even ticked up 0.021→0.023 (noise)](https://github.com/SathvikNayak123/DeepResearch/pull/2#issuecomment-4873674479) |
| [#3](https://github.com/SathvikNayak123/DeepResearch/pull/3) | `demo/ci-gate-broken` | [run 9](https://github.com/SathvikNayak123/DeepResearch/actions/runs/28644714256) | **failure** (job exits 1, as designed) | [FAIL — `frames.task_completion_rate` and `musique.task_completion_rate` both dropped 100 points (1.000→0.000); `frames.accuracy` -15, `frames.citation_precision` -70 (both judge-scored side effects of every run raising `BudgetExceeded`, not the metric this break specifically targets)](https://github.com/SathvikNayak123/DeepResearch/pull/3#issuecomment-4873674479) |

Exactly the designed outcome: bootstrap and clean both green with a real
before/after/delta table in the PR comment; broken red with
`task_completion_rate` — the metric added specifically because it's immune
to `FakeLLMClient`'s judge-RNG and F1-floor blind spots — named in the
failure output on both benchmarks. PR #3 is left **open-then-closed,
unmerged** as the retained evidence artifact per this session's brief; #1
and #2 are left open for the user to merge at their discretion.

# trace-replay dogfood: a real MuSiQue run, a real crash, and a real gap (2026-07-05)

An external project, [trace-replay](https://github.com/SathvikNayak123/trace-replay) (deterministic
agent-run replay, built on a sibling project ctx-capture's trace schema), used this repo as its v1
dogfood case study — real `ANTHROPIC_API_KEY` set in `.env` for the occasion, 3 real MuSiQue
questions run through `eval.run_eval` (not `FakeLLMClient` — the first non-zero-cost runs in this
repo's history). Two findings, one dead end, all real:

1. The self-hosted `bge-reranker-v2-m3` cross-encoder crashes deterministically on first use in
   that environment (`NotImplementedError: Cannot copy out of meta tensor...` — a
   `sentence-transformers`/`transformers`/`torch` version incompatibility, distinct from the
   already-documented slow-download CI hang above). Confirmed reproducible by re-running the same
   question in isolation.
2. That crash turned out to be **structurally undebuggable via replay**, because
   `RunRecorder` (`src/deepresearch/store/recorder.py`) is an in-memory batch accumulator flushed
   only at the end of a successful run — a crash anywhere discards every trajectory/tool-call row
   for that run, including stages that already completed. There was nothing to import for
   trace-replay to resume from. A concrete, measured argument for instrumenting incrementally
   (ctx-capture-style, per-step) rather than batching at the end, if per-step debuggability across
   crashes matters here.
3. A different, completed real run (the "Ratata" MuSiQue question) got its actual gold answer
   right but synthesized an honest gap about a secondary entity's date of birth. trace-replay
   resumed from the one worker stage that produced that gap, with a differently-phrased
   sub-question, for $0.00995 against $0.084 for the full original pipeline — same null result,
   independently confirmed against the raw corpus as a genuine coverage gap rather than a query
   quality problem.

Full writeup, real numbers, and the exact replay configuration:
[trace-replay's `docs/CASE_STUDY.md`](https://github.com/SathvikNayak123/trace-replay/blob/main/docs/CASE_STUDY.md).
Nothing in this repo's own code changed as a result — this section exists purely as the
cross-link the case study asked for.

# Real-model baseline: Gemini 2.5 Flash via OpenRouter (2026-07-05)

The prior real-key attempt (trace-replay dogfood, above) used `claude-opus-4-8` and ran out of
Anthropic credit after 3 questions; separately, a dedicated real-key `eval-smoke` attempt on
`claude-opus-4-8` (see `docs/proof/real_agentic_trajectories.json`) ran out of credit at 18/20
FRAMES questions with no aggregate score ever computed. This is the first **complete** n=20+n=20
FRAMES+MuSiQue real-model run in this repo's history — no `FakeLLMClient`, no credit exhaustion.

**Why a different model**: Anthropic credit was exhausted (confirmed via a live `400
invalid_request_error: credit balance too low`) and AWS Bedrock was blocked by an AWS Marketplace
payment/e-mandate issue on the account attempting it. `src/deepresearch/llm/client.py` already had
a `DEEPRESEARCH_LLM_PROVIDER=bedrock` swap from earlier in this session; this run added a third
path, `openrouter` (OpenAI-compatible chat-completions API — a genuinely different request/response
shape from Anthropic's Messages API, not just a different bill for the same model), verified
against OpenRouter's own docs before use. Model: `google/gemini-2.5-flash` for all four agent
stages, `google/gemini-2.5-flash-lite` for the judge (FRAMES only — MuSiQue is scored by string-based
Answer F1, no judge calls). Pricing verified live 2026-07: $0.30/$2.50 per MTok in/out for Flash,
$0.10/$0.40 for Flash-Lite — both cheaper than `claude-haiku-4-5`'s $1/$5, let alone
`claude-opus-4-8`'s $5/$25.

**Config** (from the `runs` table, `git_sha f2a0a66775101afa96dd7a5199229d0fd4a68a0a`):
```json
{
  "search_backend": "local_corpus",
  "cache_enabled": false,
  "rerank_enabled": false,
  "max_workers": 4,
  "coverage_threshold": 0.8,
  "planner_model": "google/gemini-2.5-flash",
  "worker_model": "google/gemini-2.5-flash",
  "reflection_model": "google/gemini-2.5-flash",
  "synthesis_model": "google/gemini-2.5-flash",
  "judge_model": "google/gemini-2.5-flash-lite"
}
```
Run store: a dedicated `sanity_check.db` (not `deepresearch.db`/`ci_baseline.json`) — kept
separate deliberately, since a different model family isn't a like-for-like comparison against the
committed FakeLLMClient baseline.

## Results

| Benchmark | n | accuracy / answer_f1 | citation coverage / precision | task_completion_rate | agent cost | judge cost | wall-clock |
|---|---|---|---|---|---|---|---|
| FRAMES | 20 | 0.35 | 0.483 / 0.515 | 1.00 | $0.115 | $0.0021 (51 judge calls, 0 cache hits) | 236s |
| MuSiQue | 20 | 0.077 (answer_contains_gold 0.6) | — | 1.00 | $0.112 | — | 208s |

Total real spend: **$0.229** for a complete 40-question run — confirmed against OpenRouter's own
`/api/v1/key` usage counter (delta matched the sum of reported costs, modulo a small earlier
1-question sanity probe run separately).

## Reading this honestly

- **FRAMES accuracy of 0.35 is real signal, not a regression.** The committed `ci_baseline.json`
  entry of `frames.accuracy = 0.700` is `FakeLLMClient`'s judge returning `rng.random() < 0.7` —
  disclosed elsewhere in this file as having zero dependence on actual content. This 0.35 is the
  first real measurement either baseline has ever had; the two numbers are not comparable and
  this run is **not** wired into the CI gate or `ci_baseline.json`.
- **This is one model family at n=20, not a benchmark of "which model is best."** Gemini 2.5 Flash
  is a fast/cheap tier, not Google's frontier model; no Claude-vs-Gemini accuracy comparison is
  possible here since no Claude run ever completed at this n (see the credit-exhaustion history
  above) — only cost and mechanics are directly comparable.
- **task_completion_rate 1.00 on both confirms the pipeline is provider-agnostic in practice**,
  not just in code: real decomposition, real worker/reflection/synthesis round-trips, real
  structured-JSON-schema parsing, all through OpenRouter's OpenAI-compatible endpoint rather than
  Anthropic's Messages API, with zero exceptions across 40 questions.
- **What this doesn't test**: only one non-Claude model tried; no reliability-job repeats (variance
  unknown) on this model/provider combination; FRAMES citation precision (0.515) hasn't been
  spot-checked by hand against the underlying corpus the way the original rerank ablation's
  supporting-paragraph positions were.

## Reproducing

```bash
# .env: DEEPRESEARCH_LLM_PROVIDER=openrouter, OPENROUTER_API_KEY=..., and
# DEEPRESEARCH_{PLANNER,WORKER,REFLECTION,SYNTHESIS,JUDGE}_MODEL set to
# google/gemini-2.5-flash / google/gemini-2.5-flash-lite as above
pip install -e ".[dev,eval]"
python -m eval.run_eval --mode smoke --database-url "sqlite+aiosqlite:///./sanity_check.db"
```

# Real CI baseline, real Postgres, real Docker: the first fully-clean 40-question run (2026-07-07)

Everything above this section either used `FakeLLMClient` or (the one exception) a separate
`sanity_check.db` deliberately kept out of `results/ci_baseline.json`. This session had, for the
first time, all three real prerequisites at once: a paid `OPENROUTER_API_KEY` with headroom, a
running Docker daemon (real Postgres + Redis, not SQLite/fakeredis fallbacks), and rerank left ON
(previous real-key runs disabled it to dodge a crash — see below for what that crash actually was).
`results/ci_baseline.json` now holds real, non-RNG numbers for the first time in this project's
history.

## Three real bugs, found only because real infra was finally running

None of these were reachable by `FakeLLMClient` or by any prior session's sandbox (no Docker, no
persistent paid key) — they only surfaced once the actual production code paths ran for real.

1. **Rerank concurrency race → crash.** `CrossEncoderRerankBackend`'s lazy model load
   (`src/deepresearch/rerank/bge.py`) had no lock. `orchestrator.py` builds one rerank backend
   instance per run and shares it across the bounded worker pool — the first real multi-worker
   question fired several concurrent `rerank()` calls that raced into `CrossEncoder(...)`
   construction simultaneously, corrupting `transformers`' meta-device init state
   (`NotImplementedError: Cannot copy out of meta tensor`). Reproduced directly (4 concurrent
   `rerank()` calls on a fresh instance, deterministic crash) and fixed with double-checked
   `asyncio.Lock`ing around the one-time load. This is almost certainly the same crash the
   trace-replay dogfood session hit and described as "a `sentence-transformers`/`transformers`/
   `torch` version incompatibility" — it wasn't a version problem, it was this race, just easy to
   misattribute since it's version/timing-sensitive (torch/transformers versions affect how
   loudly the corrupted state fails).
2. **CPU oversubscription once the race was fixed.** Serializing the *load* wasn't enough — a real
   FRAMES question with `max_workers=4` still measured **391s mean rerank latency**, ~28x the
   isolated single-call ablation's 13.8s mean (this doc, rerank ablation section), enough to trip
   the 600s wall-clock budget ceiling (`wall_clock_exceeded: 808.5s > 600.0s`, a genuine stopping-
   criterion firing under real load, docs/DESIGN.md decision row 3 working as designed). Root
   cause: each `CrossEncoder.predict()` call internally claims all CPU cores for its own
   BLAS/torch threads; four workers doing that concurrently thrashes instead of scaling.
   Serializing actual inference (not just the load) via a second `asyncio.Lock` fixed it —
   confirmed directly: 4 concurrent 140-chunk calls went from 391s/call (contended) to a
   consistent ~130s/call (serialized), and real re-run of the previously-failing question no
   longer hit the ceiling.
3. **Redis RESP3/HELLO incompatibility.** `redis-py` 8.x defaults to negotiating RESP3 via a
   `HELLO` command on connect; `redis:7-alpine` (docker-compose's `redis` service) rejects it
   (`ResponseError: unknown command 'HELLO'`). This is in the **live, default-cache-enabled**
   production path (`src/deepresearch/backends/__init__.py`) — every prior session had no Docker
   daemon, so `DEEPRESEARCH_CACHE_ENABLED=true` (the default) against a real Redis had never
   actually been exercised until now. Fixed with an explicit `protocol=2` on client construction
   (confirmed: `redis-cli PING` → `PONG`, server version 7.4.8, fully RESP3-capable — this is a
   client-library default mismatch, not a broken server). Verified against the real
   `build_search_backend()` path directly (a real cache miss then hit, `$0.008` correctly
   attributed as saved) and via `scripts/cache_measurement.py --n 20` now genuinely running against
   real Redis (`used_real_redis: true` in its output, not the fakeredis fallback every prior
   session's cache measurement used).

## The fix that mattered most: capping chunks-per-source before reranking

The concurrency fixes above removed *contention* but not the underlying cost: FRAMES' full
Wikipedia articles chunk into dozens of ~800-char windows (this doc's earlier finding: 146.5 mean
candidates/call vs. MuSiQue's ~7), and every chunk got scored regardless of how many survive
reranking (`rerank_top_k=6`). Added `cap_chunks()` (`src/deepresearch/chunking.py`) — an
evenly-spaced subsample to `max_chunks_per_source` (default 10, `DEEPRESEARCH_MAX_CHUNKS_PER_SOURCE`)
applied per source before the candidate pool is built, a no-op for MuSiQue's already-short
pre-chunked paragraphs.

Effect, same previously-slow FRAMES-20 subset, before/after (both post-concurrency-fix):

| | Before cap | After cap |
|---|---|---|
| Wall-clock (n=20) | did not finish cleanly under a lighter fix attempt (see below) | 2029.9s (~101s/question) |
| `task_completion_rate` | 0.95 (1 `budget_exceeded`, uncapped chunks) | **1.00** |
| `frames.accuracy` | 0.40 | **0.45** |
| `frames.citation_coverage` | 0.671 | **0.830** |
| `frames.citation_precision` | 0.688 | **0.904** |

The citation-quality jump is the more interesting result — not just "faster," genuinely *better*.
Evenly-spaced sampling gives all 6 fetched sources fair representation in the pool the reranker
sees; before the cap, one long article's sheer chunk volume could dominate the top-6 selection by
redundancy alone, starving the other 5 sources. This is consistent with, not contradicting, the
original rerank ablation's finding (rerank quality delta is real and large) — it's a second,
independent finding about candidate-pool *composition*, not the reranker itself.

## `answer_f1` vs. `answer_f1_extracted`: the raw metric was measuring report length, not correctness

MuSiQue's gold answers are short (a name, date, number). This agent produces a full cited
multi-sentence report. Standard SQuAD-style token F1 computed directly against the raw report text
crushes precision purely from length — a real example from this run, run_id `2d5c95ad...`:

> Report: *"...Springfield became the capital of Illinois in 1839 [src_25]. This change was largely
> influenced by Abraham Lincoln and his colleagues..."*
> Gold: `"1839"` → `answer_f1 = 0.05`, despite `answer_contains_gold = 1`.

Added `Judge.extract_short_answer` (`eval/judge.py`, `eval/prompts/extract_answer_v1.txt`) — a
cheap judge-model call (same cache/cost-tracking pattern as `judge_accuracy`/`judge_citation`) that
pulls the terse stated answer out of the report before scoring. Wired into `run_musique()` as an
**additional** metric, `answer_f1_extracted` — the original `answer_f1` (raw report vs. gold) is
kept unchanged and still gated in `ci_baseline.json`/`ci_gate.py`, since redefining an already-
baselined metric's meaning in place would silently invalidate history. `answer_f1_extracted` is now
also gated (`eval/ci_baseline.py`'s `_QUALITY_METRICS`).

Backfilled for this run's already-completed 20 MuSiQue questions from their stored report text
(no agent re-run needed — 20 extraction calls, $0.00066 total):

| Metric | Value |
|---|---|
| `answer_f1` (raw report) | 0.059 |
| `answer_f1_extracted` | **0.490** |
| `answer_contains_gold` | 0.6 |

`answer_f1_extracted` sits below `answer_contains_gold` as expected (exact/partial phrase-overlap F1
is a strictly harder bar than substring containment — a few reports hedged without committing to an
answer, extracted as `""`, scoring 0). **`answer_f1_extracted` is the number to cite against
MuSiQue's own published (short-answer) baselines going forward** — `answer_f1` alone was never a
fair comparison for a long-form-report architecture and is kept only for historical/regression
continuity.

## Final real baseline (`results/ci_baseline.json`, git SHA `84201253`)

| Metric | Value |
|---|---|
| `frames.accuracy` | 0.450 |
| `frames.citation_precision` | 0.904 |
| `frames.task_completion_rate` | 1.000 |
| `frames.cost_per_query_usd` | $0.0056 |
| `musique.answer_f1` | 0.059 |
| `musique.answer_f1_extracted` | 0.490 |
| `musique.task_completion_rate` | 1.000 |
| `musique.cost_per_query_usd` | $0.0056 |

`scripts/ci_gate.py` run against this same data confirms self-consistency (all 7 gated metrics
`OK`, zero delta). Total real spend across both re-runs and the backfill: ~$0.35.

## Real reliability (20q x 3 repeats), and a fourth real bug it surfaced

```bash
python -m eval.run_eval --reliability --n 20 --repeats 3 --database-url "$PGURL"
```

All 60 question-runs (real agent execution, real cost, $0.3288 total) completed and were stored
successfully — then the run crashed on its very last step, writing the 3 aggregate summary metrics.
**Fourth real bug found only against real Postgres**: `run_reliability` anchored those summary rows
to a synthetic `run_id` string, `f"reliability-{git_sha}"` (52 characters) — but `eval_scores.run_id`
is a native Postgres `UUID` column with a foreign-key constraint to `runs.run_id`. That string is
neither UUID-shaped nor a real `runs` row. SQLite's loose typing (a `String(36)` column happily
stores any string of any length) and its FK-enforcement-off-by-default both masked this in every
prior sandbox session; Postgres rejected it immediately (`invalid UUID`). Fixed by creating a real,
dedicated `runs` row (`benchmark_name="reliability"`, a proper `config` dict) to anchor the summary
metrics to — the same pattern every other real question already uses, just applied to the
aggregate-level write this one function had skipped.

The 60 already-completed, already-paid-for question-runs didn't need re-running: reconstructed
`per_question_correct` by matching the 60 stored `runs` rows (ordered by `created_at`) back to
`musique_bench.load_subset(20, 42)`'s deterministic 3x-repeated order, pulling each run's already-
stored synthesis report text and re-scoring with `gold_contained`. The recovered numbers matched
the original crashed run's own (partially-visible-in-traceback) computed values exactly, confirming
the reconstruction is exact, not an approximation.

| Metric | Value |
|---|---|
| Mean accuracy | 0.533 |
| Stdev accuracy | 0.024 |
| All-consistent (pass^k) rate | 0.60 |

First real (non-`FakeLLMClient`) reliability measurement in this project's history. 60% of the 20
questions got the same `gold_contained` verdict across all 3 repeats; the other 40% flipped at
least once — with a real model, that's genuine signal that those questions are borderline for the
agent (ambiguous phrasing, a source the reranker sometimes surfaces and sometimes doesn't), not
benchmark noise, unlike every prior reliability run in this repo which used `FakeLLMClient`'s
independent-random-draw judge and could only validate the variance-reporting *mechanics*, not
produce a real signal.

## Architecture ablation, real accuracy columns — a fifth real bug, and a real reversal signal

```bash
python scripts/architecture_ablation.py --n 20 --seed 42 --database-url "$PGURL"
```

**Fifth real bug**: `plan_first_pool1` crashed on a real, sustained OpenRouter 429 (`"JSON error
injected into SSE stream"`) that outlasted the LLM client's 3-attempt/~4.5s retry window added
earlier this session — the first occurrence (this session's smoke run) was a one-off blip that 3
quick retries absorbed; this one, under heavier same-day call volume, didn't clear that fast.
Hardened `_complete_json_openrouter`'s retry to 5 attempts / 2-4-8-16s backoff (~30s total) —
re-ran clean on the next attempt. The partial first attempt's rows (20 `plan_first_pool4` + 7
partial `plan_first_pool1`, one left stuck `running` by the crash) were deleted before dumping
final numbers, same cleanup pattern as the FRAMES re-run above.

## Results (n=20, MuSiQue, real Gemini 2.5 Flash, rerank on)

| Variant | s/question | tokens/task | steps/task | `answer_f1` | `answer_contains_gold` | cost (20q) | iterations |
|---|---|---|---|---|---|---|---|
| `plan_first_pool4` (default) | 40.1 | 8370 | 42.8 | 0.055 | 0.45 | $0.104 | 2.4 |
| `plan_first_pool1` | 57.5 | 9282 | 47.2 | 0.045 | 0.45 | $0.119 | 2.5 |
| `react` | 43.9 | **5468** | 27.2 | **0.067** | **0.50** | **$0.074** | 3.4 |

Full config and raw data: `results/architecture_ablation_20260707T130241Z.json`.

## Reading this honestly

**Row 1 (worker-pool size): the first real evidence that parallelism actually pays for itself.**
The `FakeLLMClient`-era sweep (docs/DESIGN.md §10) found `pool1` and `pool4` "almost identical" —
an artifact of the fake client's instant responses masking any real wall-clock benefit from running
workers concurrently. With real per-call network latency, `pool4` is ~30% faster than `pool1`
(40.1s vs. 57.5s/question) at essentially the same or slightly better accuracy and lower cost. This
doesn't reverse row 1's decision — it's the first measurement that actually *validates* it, on real
latency rather than a structural argument alone.

**Row 2 (planning style, the flagship ablation): real evidence pointing toward, but not yet meeting,
the row's own stated reversal bar.** DESIGN.md row 2 names its reversal condition precisely: *"if
ReAct matches plan-first on accuracy/citation-precision at lower cost... default switches to
ReAct."* On this run, `react` doesn't just match plan-first's accuracy — it's directionally *ahead*
on both `answer_f1` (0.067 vs. 0.055) and `answer_contains_gold` (0.50 vs. 0.45), while using 35%
fewer tokens and costing 29% less. The `FakeLLMClient`-era finding (react costs ~1.9x plan-first's
tokens, mechanically, one extra round-trip per query decided) was correct as a structural argument
but apparently doesn't dominate in practice: real react conversations settle in fewer total steps
(27.2 vs. 42.8) than a plan-first run that decomposes into more sub-questions than turn out to be
needed, more than offsetting the extra per-step round-trip cost.

**What keeps this from being a clean decision-reversal, stated plainly**: this is one n=20 run per
variant, not a repeated-run comparison — CLAUDE.md's own rule ("reliability evals repeat a subset
3-5x and report variance, not a single score") wasn't applied to *this* ablation, only to the
default-config reliability job above. The accuracy gaps (0.067 vs. 0.055; 0.50 vs. 0.45) are
directionally consistent across both metrics, which is a stronger signal than either alone, but
still small enough on n=20 that a repeat-3x version of this exact ablation could plausibly narrow
or widen the gap. Wall-clock also still favors `plan_first_pool4` (40.1s vs. react's 43.9s) despite
react's lower token count — react's sequential-by-construction design (no worker pool: "one query
decided at a time," docs/DESIGN.md row 2) can't parallelize the way plan-first's bounded pool can,
so real per-request latency for a live query still favors the current default.

**Recommendation, not a decision**: this is real, promising, decision-reversing-shaped evidence —
enough to justify a repeated (3-5x) version of this same ablation as the next concrete step before
actually flipping the default, not enough on its own to flip it today. Flagging for the user's call
rather than changing `RunConfig.planning_style`'s default unilaterally.

# Session 6: live streaming, thin UI, and a first real observability verification (2026-07-07)

## Streaming API + thin UI

Added `GET /research/stream` (SSE — GET, not POST, so the browser's native `EventSource` needs no
custom client code) and `GET /runs/{run_id}` (`src/deepresearch/api/streaming.py`,
`routes_runs.py`). `orchestrator.py`'s `run_research()` gained an optional `on_event` hook, fired
inside `_call_stage` right after each stage's trajectory is recorded, plus one `run_started` event
at the very top (before any DB/LLM work) so a live client sees the `run_id` — and therefore the
Langfuse trace ID — from the first byte, not only at completion.

A test written against this hook (`test_run_research_on_event_fires_for_every_stage`) initially
failed: the "worker" stage never fired. Root cause was in my own edit, not pre-existing code — a
`replace_all` edit meant to wire `on_event` into both worker call sites (the sequential ReAct-mode
one and the parallel-pool `_bounded_worker` one) only matched one, because the two blocks have
different indentation and `replace_all` only replaces literal string matches, not both semantically
equivalent occurrences. Worth recording because it's exactly the kind of self-introduced bug a
dedicated test catches immediately and a manual click-through wouldn't have.

A second real bug surfaced by manual `curl` testing against a live server (not caught by the
initial unit tests, since those never exercised the endpoint's own request-handling code): a bad
`local_corpus_dir` in the `config` query param crashed with a raw 500 instead of a graceful
`run_error` SSE event, because `RunConfig.from_overrides(...)` and `build_search_backend(...)` ran
*before* `StreamingResponse` was constructed — any exception there propagated as an unhandled
FastAPI error, never reaching the try/except inside the streaming generator. Fixed by moving both
calls inside the generator itself, so every failure mode (malformed config JSON, a missing corpus
file, an agent-side exception) degrades to the same `run_error` event. Regression-tested in
`tests/test_streaming.py` (4 tests: bad-config, malformed-JSON, mocked-success-path event ordering,
mocked-agent-exception) — the success-path test mocks `run_research` itself rather than hitting a
real LLM, since this test is about the endpoint's own queue/SSE plumbing, not agent behavior
(already covered separately in `test_orchestrator_persistence.py`).

Thin UI at `/ui/` (`ui/index.html`, mounted via `StaticFiles` with a path resolved relative to
`main.py`'s own file location, not `cwd` — works identically from a local dev shell, Docker's
`/app`, or a test runner's directory, and `check_dir=False` so a deployment that never shipped
`ui/` 404s on `/ui/*` instead of crashing the whole app at import time). Type a question, watch the
trace populate stage-by-stage, read the cited report when `done` fires.

**Verified against a real, running HTTP server** (not just FastAPI's `TestClient` ASGI transport):
started `uvicorn` for real, `curl`'d the full SSE stream for a real question, confirmed correct
event ordering across a real 3-round replan loop (`run_started` → `stage_complete`×N → `done`),
confirmed `GET /runs/{run_id}` retrieves that exact run's 11 trajectory rows + 48 tool-call rows,
and confirmed the `run_error` fix live (a bad corpus path now yields one `run_error` SSE frame,
not a 500).

## Observability verification: two previously-unverified claims, now confirmed real

**Langfuse trace-by-`run_id`** (docs/DESIGN.md §6's own flagged open risk: *"OTel context/baggage
propagation across truly concurrent async workers... verify it holds before relying on it
architecturally"*) — queried Langfuse Cloud's public API directly (`GET /api/public/traces/{id}`)
for the `run_id` from the live streaming test above. **Confirmed**: the trace's `id` field is
exactly that `run_id`, with 18 real observations — `plan`, 6×`worker`, 6×`rerank` (nested under
their respective worker spans), 3×`reflection` (matching the 3 real replan rounds), `synthesis`,
and the root `run` span — every parent/child relationship intact. This is the first time this
project has actually checked whether the claimed `run_id = trace_id` join works under real
concurrent worker load rather than assuming it from the OTel SDK's documented behavior. It does.

**Grafana cache-hit-rate dashboard against real traffic** — brought up `app` + `prometheus` +
`grafana` via `docker compose up -d --build` (deliberately *not* the self-hosted Langfuse tier,
since `.env` points `LANGFUSE_HOST` at Langfuse Cloud instead — no reason to run ClickHouse/MinIO
locally for a service that's unused). Made two real live `POST /research` calls (real Tavily search
+ real OpenRouter LLM, the actual production path, not `local_corpus` — the Docker image
deliberately doesn't ship `data/corpus/`, that's eval-only data, so a `local_corpus` config against
the container 404s on the corpus file; confirmed live, not a bug, just the wrong test setup on my
first attempt). Result: real cache hits (`fetch: 5 hits, 7 misses` from the two overlapping
questions), Prometheus scraping `deepresearch-app: up`, and the dashboard's own literal panel
queries confirmed against Prometheus's query API — the cumulative-count panel renders correctly
immediately; the `rate(...[5m])`-based hit-rate-percentage panel returns `0` right after a two
request burst, which is expected `rate()` behavior needing more than one scrape's worth of history
to compute a meaningful per-second slope, not a broken panel. Not verified: sustained traffic over
a longer window (which would need many more real API calls than this session's cost budget called
for) — the mechanics are confirmed real, the multi-request behavior is a reasonable, not fully
load-tested, extrapolation.

## `eval-full` (100q FRAMES + 100q MuSiQue): complete

Kicked off after the Session 5 chunk-cap fix made the ~2-2.5hr FRAMES-full estimate from the
evals-as-a-system session plausible to actually run rather than defer again. Interrupted ~24
questions into FRAMES by an unrelated Claude Code session restart (not a code bug — the harness
process itself exited); restarted clean rather than building a custom resume script for ~$0.15 of
sunk cost. Ran to completion on the second attempt with zero further crashes.

```bash
python -m eval.run_eval --mode full --database-url "$DATABASE_URL"
```

Full output: `results/eval_full_20260707T182512Z.json`.

| Benchmark | n | Wall-clock | Task completion | Key metric(s) | Agent cost | Judge/extraction cost |
|---|---|---|---|---|---|---|
| FRAMES | 100 | 9264.5s (~2.57h) | 1.00 | accuracy 0.460, citation coverage 0.649, citation precision 0.785 | $0.617 | $0.0147 (371 calls, 13 cache hits) |
| MuSiQue | 100 | 3894.4s (~1.08h) | 1.00 | `answer_f1` 0.056 (raw), **`answer_f1_extracted` 0.347**, `answer_contains_gold` 0.44 | $0.619 | $0.0034 (100 extraction calls) |

Total real cost for the full 200-question suite: **~$1.25**. Zero exceptions, zero budget-ceiling
hits across all 200 real questions — a meaningfully larger, more stable sample than the n=20 smoke
baseline (`results/ci_baseline.json`), consistent with it directionally (smoke: `frames.accuracy`
0.450 / `citation_precision` 0.904; full-100: 0.460 / 0.785 — accuracy holds up, citation precision
comes in a bit lower at 4x the sample, a real signal worth more trust than the smaller run alone).

**A cleanup mistake, disclosed rather than quietly fixed**: consolidating stray rows from the
interrupted first attempt, a timestamp-boundary guess (misremembered as "07:36-08:50 UTC" instead
of the actual "10:31-11:03 UTC") caused the delete filter to also catch the original Phase-1 clean
20-question FRAMES smoke-baseline rows — the ones `results/ci_baseline.json` was computed from.
**Nothing published or committed was affected**: `ci_baseline.json` is a static, already-frozen
JSON snapshot (git-tracked), and this document's own write-up of that baseline is prose, not a live
query — neither depends on the DB rows still existing. What's actually gone: the ability to
re-query that specific historical batch's raw per-question `trajectories`/`tool_calls` from the
local Postgres dev database (no backup existed; this was never pushed anywhere durable). Verified
before deleting that `musique`'s equivalent old baseline rows were correctly excluded (they were,
confirmed by exact post-delete row count) — the mistake was scoped to `frames` only. Recorded here
in the same spirit as this document's other real mistakes (the reliability job's invalid run_id,
the DRB flush-at-end-of-loop bug): the honest record includes what went wrong, not just what
worked.

### Reading this honestly

- **`musique.answer_f1_extracted` (0.347) at n=100 is meaningfully lower than the n=20 backfilled
  value (0.490) from the smoke baseline.** Both are real signal, not a regression — n=20 was always
  a small, noisier sample; 0.347 at 5x the sample size is the more trustworthy number going
  forward. This is exactly the kind of thing a larger real run is for.
- **FRAMES wall-clock (2.57h for 100q, ~93s/question average) matches the post-chunk-cap-fix
  smoke rate closely** (~101s/question at n=20) — confirms the fix's effect holds at scale, not
  just on the small sample it was validated against.
- **What this doesn't test**: no reliability/variance repeats at this n (CLAUDE.md's own rule
  wasn't re-applied at the full-100 scale, only at n=20 earlier); this real baseline was not used to
  re-tighten `ci_baseline.json`'s gated thresholds — that's a reasonable next step now that a larger
  real sample exists, not done in this session.

## DeepResearch Bench: real RACE + FACT-style scoring, minimal proof-of-mechanics (n=2)

Replaced the `NotImplementedError` stub (docs/DESIGN.md decision row 11) with a real
implementation, adapted from `Ayanami0730/deep_research_bench` (MIT), pinned commit
`469cce54ea7f6a63c163d3d9fec879cf289ec484`. Two honest departures from the reference, both
documented in the module docstring (`eval/benchmarks/deepresearch_bench.py`):

- **RACE** is a faithful port: real per-task criteria (fetched + cached from the reference repo's
  `criteria.jsonl`, each dimension pre-weighted), the real point-wise scoring prompt structure, and
  the real weighted-aggregation math from their `utils/score_calculator.py` (`eval/race_judge.py`).
  Judge model is this project's own configured `judge_model` (`google/gemini-2.5-flash-lite` via
  OpenRouter), not the reference's GPT-5.5 — scores here are not comparable to the public
  leaderboard, which is scored under one fixed evaluator.
- **FACT** in the reference implementation extracts claim-URL pairs from free text and independently
  re-scrapes each URL via Jina (a dependency this project has no key for) to verify support. This
  agent already produces structured `claim -> source_id` citations against already-fetched content,
  so this reuses the existing, already-tested `compute_citation_metrics` (FACT-protocol-style per
  CLAUDE.md) instead of building a second, redundant re-scrape pipeline. Methodologically
  equivalent in spirit, not a byte-exact FACT port, not leaderboard-comparable either.

Ran against **live Tavily search** (`search_backend="tavily"`), not `LocalCorpusBackend` — DRB tasks
are open-ended real-world research questions with no fixed corpus, unlike FRAMES/MuSiQue. Rerank
was disabled for this run specifically (`DEEPRESEARCH_RERANK_ENABLED=false`), not as a permanent
default — `eval-full`'s FRAMES-100 was running concurrently and is CPU-bound on the reranker;
running DRB's own reranker calls at the same time would have reintroduced the exact cross-process
CPU contention Session 5 fixed. This is a smaller quality tradeoff for a Tavily-backed run than for
FRAMES specifically, since `candidate_pool_size == rerank_top_k == 6` makes it a no-op for
*selection* either way, and Tavily's results already carry a relevance ranking (unlike FRAMES' raw
local corpus, which has none).

### Two real bugs found running this for the first time

1. **Root-credentials-adjacent oversight, corrected before any cost**: my first launch attempt built
   `RunConfig` with rerank left on, which would have contended with the concurrently-running
   `eval-full`. Caught before any real cost was incurred and killed via the proper `TaskStop`
   tool — a raw `kill -9` on a `ps`-matched PID was correctly blocked by this session's own
   auto-mode safety classifier, since a PID found by pattern-matching isn't confirmed to be a
   session-owned process. Re-launched with `DEEPRESEARCH_RERANK_ENABLED=false` instead.
2. **The same "flush at the end of the loop" bug this project already fixed elsewhere, freshly
   reintroduced in this new module.** `run_musique`/`run_frames` both flush `eval_scores` per
   question specifically so a mid-loop crash can't discard already-computed, already-paid-for
   scores (their own inline comments say so explicitly). My first `deepresearch_bench.py` draft
   batched the flush to the end of the loop instead — and proved the point the hard way: a
   **sustained** OpenRouter 429 (the same failure mode from the architecture ablation, but this
   time outlasting even the hardened 5-attempt/~30s retry — confirmed reproducible, not a one-off)
   crashed mid-loop on the second task, and task one's fully-computed RACE + citation scores
   (already paid for) were silently lost — confirmed via a direct `eval_scores` row count of zero
   immediately after the crash. Fixed by moving the flush inside the loop, matching the established
   pattern. The already-completed agent runs weren't wasted: reconstructed a `RunResult` for each
   from its stored `trajectories` rows (`synthesis` stage → `Report`, `worker` stage rows → the
   exact `WorkerNotes` Pydantic objects, since that's literally how they were serialized in the
   first place) and re-ran only the judge scoring against the recovered data — paying for judging
   again, not for re-running two more rounds of live multi-hop web research.

### Results (n=2, real EN tasks, live Tavily search, real judge scoring)

| Task | Topic | RACE total | comprehensiveness | insight | instr. following | readability | citation coverage | citation precision |
|---|---|---|---|---|---|---|---|---|
| 76 | Health (gut microbiota) | 7.67 | 7.65 | 6.30 | 9.50 | 7.60 | 1.000 | 0.938 |
| 74 | Education & Jobs (sports IDSS) | 2.83 | 2.70 | 1.90 | 3.95 | 3.85 | 0.733 | 1.000 |

Real cost: **$0.038 total** ($0.0348 agent execution across both tasks + $0.0028 judge scoring) —
dramatically below the reference implementation's $15-35-for-10-questions estimate, because this
run uses cheap OpenRouter-routed models throughout rather than GPT-5.5/GPT-5.4-mini. Scaling
linearly, a real 10-question weekly run would land near **$0.19**, not $15-35 — though this is one
data point at n=2, not a stable per-task average, and task 74's much shorter report (1833 vs. 3819
chars) suggests real per-task cost variance that a larger sample would characterize better.

### Reading this honestly

- **The RACE score spread (7.67 vs. 2.83) looks like real signal, not noise.** Task 74's report is
  less than half the length of task 76's, and the low `insight`/`comprehensiveness` sub-scores
  (1.90, 2.70) are consistent with a report that didn't dig as deep into its (more specialized,
  "sports intelligent decision support system") topic — exactly the kind of differentiation RACE's
  weighted-criteria design is meant to produce, not an artifact.
- **Citation coverage/precision moved in opposite directions between the two tasks** (76: high
  coverage/high precision; 74: lower coverage/perfect precision on what *was* cited) — a small,
  real illustration of why this project tracks both metrics separately rather than collapsing them
  into one number (docs/DESIGN.md decision row 6).
- **n=2 is a mechanics proof, not a characterization of this agent's DRB-style report quality.**
  Scaling to the real weekly-10 or monthly-100 subsets is the natural next step and needs no new
  code — `--n` is already a free parameter — just more real cost and wall-clock time.
- **What this doesn't test**: no repeated-run variance (CLAUDE.md's reliability rule wasn't applied
  here, same gap as the architecture ablation); no comparison against the reference GPT-5.5/GPT-5.4-mini
  judges to see how much judge-model choice shifts these scores; FACT-style citation checking
  validates against this agent's own already-fetched content, not an independent re-scrape, so it
  can't catch a case where the fetched content itself was stale or wrong at fetch time.

### Reproducing

```bash
pip install -e ".[dev,eval]"
python -m eval.benchmarks.deepresearch_bench --mode weekly --n 2 --confirm --database-url "$DATABASE_URL"
# Real weekly-10 or monthly-100 runs: drop --n (defaults to 10/100 per --mode),
# or pass --n directly for any custom size.
```

# Architecture ablation, repeat-3x: the single-run ReAct edge does not survive (2026-07-08)

The single-run architecture ablation above ended with an explicit "not a decision" — ReAct was
directionally ahead on accuracy and clearly cheaper on n=20/single-run, but CLAUDE.md's own rule
("reliability evals repeat a subset 3-5x... never a single score") hadn't been applied to this
specific comparison. `scripts/architecture_ablation.py` gained `--repeats` and `--variants` flags
(same pass^k-style distribution machinery `eval/metrics/reliability.py` already provides) and this
is that repeat.

```bash
python scripts/architecture_ablation.py --n 20 --seed 42 --repeats 3 \
  --variants plan_first_pool4,react --database-url "$DATABASE_URL"
```

Dropped `plan_first_pool1` from this run — row 1 (parallelism) isn't in question here, only row 2
(planning style); repeating a third variant would have added ~$0.16/~11 more minutes for a
comparison this round wasn't measuring. Ran against local SQLite (Docker wasn't up at the time;
this is a standalone ablation, not shared eval history, so the dev-loop swap is a non-issue).

## Results (20q, MuSiQue, real Gemini 2.5 Flash, 3 repeats each)

| Metric | `plan_first_pool4` | `react` |
|---|---|---|
| `answer_contains_gold` (mean ± stdev) | **0.533 ± 0.126** | 0.400 ± 0.050 |
| per-repeat `contains_gold` | [0.65, 0.40, 0.55] | [0.35, 0.45, 0.40] |
| `answer_f1` (mean ± stdev) | 0.0619 ± 0.0043 | 0.0646 ± 0.0018 |
| all-consistent rate (per-variant) | 0.70 | 0.70 |
| mean tokens/question | 8643 | **5232** |
| mean cost/repeat (20q) | $0.109 | **$0.071** |
| mean wall-clock/question | 32.5s | 24.8s |

Full data: `results/architecture_ablation_20260708T0*.json` (SQLite run, not committed to the
Postgres run store used elsewhere).

## Reading this honestly

**The single-run finding reverses.** The original n=20/single-run comparison had `react` ahead on
`answer_contains_gold` (0.50 vs. 0.45). At 3 repeats, `plan_first_pool4` comes in **ahead by 13
points** (0.533 vs. 0.400) — the opposite direction. This is exactly the failure mode CLAUDE.md's
repeat-and-report-distribution rule exists to catch: a single run of a 20-question subset sits well
within one repeat's worth of noise, and the original comparison happened to land on a favorable
draw for react.

**`plan_first_pool4`'s own repeat-to-repeat variance is large** (0.40, 0.55, 0.65 — a 25-point
spread across 3 repeats of the *same* 20 questions, *same* config). This is real information, not
just noise to explain away: coverage-driven replanning (row 2's bounded-replan mechanism) means
plan-first's outcome on a given question depends on whether reflection judges coverage sufficient
on a given pass, which can vary run to run even at fixed temperature/model. `react`'s narrower
spread (0.35-0.45) is consistent with its simpler, non-replanning control flow. Both variants
landed on an identical 0.70 all-consistent rate at the *per-question* level, though — so this isn't
"react is more reliable," just "plan-first's mean is higher with wider swings."

**`answer_f1` shows no real difference** (0.0619 vs. 0.0646, both well within the other's stdev) —
consistent with the metric's own documented floor-bound / length-penalty issues (this doc's earlier
section on raw `answer_f1` vs. `answer_f1_extracted`), not strong evidence either way.

**Cost and tokens are the one finding that held up across both runs.** ReAct used ~40% fewer tokens
and cost ~35% less in both the single-run and the repeat-3x measurement — a stable, structural
result (one incremental decision call per query vs. plan-first's upfront decomposition), not an
artifact of a lucky draw. Wall-clock, by contrast, flipped direction between the two runs (plan-first
faster in the original, react faster here) — real-world API latency variance dominates a 20-40s/question
gap on n=20 more than either architecture's own structural properties do; not a reliable
differentiator either way with this sample size.

**Verdict, directly answering DESIGN.md row 2's own stated reversal condition** ("if ReAct matches
plan-first on accuracy... at lower cost, default switches"): ReAct does **not** match plan-first's
accuracy here — it's 13 points behind on the properly-repeated measurement, a wider gap than its
single-run *lead* was. **The default stays plan-first.** ReAct's real, repeat-confirmed cost/token
advantage is worth keeping documented as a genuine tradeoff (cheaper, less accurate on this
subset), not a reason to flip the default — row 2's own bar is accuracy parity *at* lower cost, and
that bar isn't met.

**What this doesn't test**: n=20 with 3 repeats is still a small sample for a 13-point gap with
±12.6-point variance on one side — a 5-repeat run (CLAUDE.md's upper bound) or a larger n would
narrow the confidence further, though the direction (plan-first ahead) is now the opposite of the
original single-run result, which is itself the more load-bearing finding than the exact magnitude.
No FRAMES-subset version of this same repeat-3x comparison exists yet (DESIGN.md §10's own flagged
follow-up, still open).
```

# LangGraph migration: real-model mechanics proof, n=3 MuSiQue (2026-07-17)

DESIGN.md decision row 14 (orchestrator hand-rolled asyncio → LangGraph `StateGraph`,
`src/deepresearch/agent/graph.py`) needed one real-model run to confirm the graph path
actually works end-to-end against a live provider, not just against the offline
`StubLLM`-driven test suite (`tests/test_graph_parity.py`, 81/81 passing at migration
time). This is a **mechanics proof at n=3, not an accuracy ablation** — same spirit as
this doc's earlier n=2 DeepResearch Bench proof — explicitly scoped that way to keep
real spend near-zero while both configured providers were being restored from a
zero-balance state (see below).

**A real bug this exact process caught, fixed before this run**: the first attempt at
this verification (same day, before top-up) hit `langgraph.errors.GraphRecursionError`
on a real MuSiQue question — `GraphState`'s `replans`/`step` fields were declared as
plain `int` instead of `Annotated[int, operator.add]`. A node returning `{"replans": 1}`
to mean "add one more" instead silently *overwrote* the channel to the literal value 1
every time (LangGraph's default reducer is last-write-wins, not accumulation) —
`replan_allowed` never went false, and the loop only stopped via LangGraph's own
recursion safety net, never the documented stopping criterion (DESIGN.md row 3).
Confirmed live: one real question produced 10 reflection calls before hitting the
ceiling. No `StubLLM`-driven test had caught this — the existing replan test only forced
*one* replan before letting the stub converge, never exercising a second consecutive
replan where "stuck at 1" diverges from "correctly incrementing." Fixed
(`Annotated[int, operator.add]` on both fields) and regression-tested
(`test_replan_ceiling_stops_even_when_llm_never_converges`, a stub that never converges
on its own so only the `max_replans` ceiling can stop it — verified by reverting the fix
and confirming this exact test reproduces the `GraphRecursionError`, then confirming the
fix resolves it).

**Zero-cost check run first, before spending on a real model**: a genuinely concurrent
3-worker `StubLLM` run through the LangGraph path with real OTel export to Langfuse
Cloud confirmed `trace_id == run_id` with the expected nested `run → {plan, worker×3,
reflection, synthesis}` structure (each worker with its own nested `rerank` span) —
re-verifying, for the new orchestration engine specifically, the same concurrent-context-
propagation risk DESIGN.md §6 flagged and Session 6 first confirmed for the native path.

## Method

Both configured providers (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) returned zero
funded balance earlier this session (`402 Payment Required` / `credit balance too low`,
confirmed directly against each). After the user topped up OpenRouter, confirmed usable
with a live 3-token completion call before spending on the real subset.

```bash
python -m eval.run_eval --benchmark musique --n 3 --seed 42 \
  --database-url "sqlite+aiosqlite:///./graph_verify_langgraph.db"
DEEPRESEARCH_ORCHESTRATION=native python -m eval.run_eval --benchmark musique --n 3 --seed 42 \
  --database-url "sqlite+aiosqlite:///./graph_verify_native.db"
```

Same model/config as this doc's other real runs: `google/gemini-2.5-flash` (agent
stages) / `google/gemini-2.5-flash-lite` (judge) via OpenRouter, local corpus, rerank
enabled, `max_workers=4`, seed 42, n=3 (stratified-by-hop sample, same deterministic
sampler as every other MuSiQue run in this doc — just the first 3 of it).

## Results (n=3, MuSiQue, real Gemini 2.5 Flash)

| Orchestration | task_completion_rate | mean tokens/task | mean steps/task | `answer_f1` | `answer_f1_extracted` | `answer_contains_gold` | agent cost | wall-clock |
|---|---|---|---|---|---|---|---|---|
| `langgraph` | **1.00** | 8047 | 37.3 | 0.104 | 0.933 | 0.667 | $0.0160 | 117.0s |
| `native` | **1.00** | 8287 | 45.3 | 0.044 | 0.600 | 0.667 | $0.0148 | 173.8s |

Total real spend: **$0.031** for both runs combined (agent + extraction), within the
~$0.05–0.15 the user approved for this specific check.

## Reading this honestly

- **The one number that matters most here is `task_completion_rate: 1.00` on both** —
  the graph path completed all 3 questions cleanly with zero crashes and zero
  budget/recursion exceptions, on the exact real-model path the earlier
  `GraphRecursionError` came from before the fix. That's what this run was actually
  for; it isn't sized to say anything about accuracy or cost differences.
- **n=3 is far too small to read the `answer_f1`/`answer_contains_gold` gaps as a real
  signal either way** — this doc's own repeat-3x architecture ablation (above) already
  demonstrated that even n=20 single-run comparisons on this exact metric family sit
  well within one repeat's worth of noise (the react-vs-plan-first finding fully
  reversed between a single n=20 run and a repeat-3x version). n=3 is smaller still;
  reading "langgraph scored higher" out of this table would repeat the exact mistake
  that repeat-3x section exists to warn against.
- **`answer_contains_gold` matching exactly (0.667 both)** is a mildly reassuring
  mechanics signal — both orchestration paths reached the same correct/incorrect
  verdict on all 3 questions — but is not a substitute for the real, larger,
  properly-repeated PR-smoke + ablation run DESIGN.md row 14 still calls for before
  the migration's behavior-neutrality claim is fully closed.
- **What this doesn't test**: the real n=20+20 PR smoke gate against `ci_baseline.json`,
  a repeated (3-5x) langgraph-vs-native ablation, and a reliability-subset run on the
  graph path — all still open, gated on real spend beyond what this specific proof was
  scoped to.

## Reproducing

```bash
pip install -e ".[dev,eval]"
python -m eval.run_eval --benchmark musique --n 3 --seed 42 --database-url "$DATABASE_URL"
DEEPRESEARCH_ORCHESTRATION=native python -m eval.run_eval --benchmark musique --n 3 --seed 42 --database-url "$DATABASE_URL"
```

# Phase 2, Session 1: failure-mode analysis + citation-regex fix (2026-07-18)

Requirement-2 work ("improve the scores") is gated by the project's own rule
(DESIGN.md §7): don't build a score lever before failure-mode data says which
failure it addresses. This session built that data rather than guessing at levers.

**A latent Phase-1 regression, fixed first.** `eval/metrics/citation.py`'s
`CITATION_MARKER_RE` was `\[(src_\d+)\]` — digit-only, so it silently missed the
namespaced `[src_1abe120f_1]` markers Phase 1's parallel-worker fix introduced on
the LangGraph default path. FRAMES `citation_coverage`/`precision` would have read
~0 on every graph-path run. Widened to `\[(src_\w+)\]`, regression-locked in
`tests/test_eval_metrics.py`. The real run below is the live confirmation: FRAMES
`citation_coverage=0.702`, `precision=0.758` on graph-format reports (non-zero =
fix works).

**The failure-analysis tool** (`scripts/failure_analysis.py`, read-only, no LLM):
for each missed FRAMES/MuSiQue question it recovers question/sub-questions/report/
claims from the run store, matches to gold via the deterministic samplers, re-runs
*retrieval only* (BM25 + bge rerank, no LLM) using the run's own stored config, and
classifies the miss as **retrieval_or_composition** (gold never in the reranked
chunks — either retrieval missed it or, for computed/multi-hop answers, it never
appears verbatim), **extraction** (gold was in a chunk but no worker claim captured
it), or **synthesis** (a claim captured it but the final answer is still wrong).
Validated free on `sanity_check.db` (Sonnet-5, n=20+21) before spending.

**Fresh real batch** on the current config (Gemini 2.5 Flash, LangGraph path,
`--mode smoke`, 20+20, `phase2_failure.db`): FRAMES accuracy 0.45 (identical to the
committed native `ci_baseline.json` — a Phase-1 behavior-neutrality data point),
MuSiQue `answer_contains_gold` 0.50 / `answer_f1_extracted` 0.415. `task_completion`
1.0 both. Cost **$0.233** agent+judge.

## Failure distribution (current Gemini config, `results/failure_analysis_20260718T111024Z.json`)

| Benchmark | Correct | retrieval_or_composition | extraction | synthesis |
|---|---|---|---|---|
| FRAMES | 9/20 | 9 (82% of misses) | 1 (9%) | 1 (9%) |
| MuSiQue | 10/20 | 1 (10%) | 6 (60%) | 3 (30%) |

## Reading this honestly

- **FRAMES misses are dominated by composition, not retrieval.** 82% are
  gold-never-in-any-chunk, and the sanity-check spot-check showed these are almost
  all `Numerical reasoning` questions whose gold answer is a *computed* value ("14
  years younger", "43,518") that no source states verbatim. **Graph retrieval /
  better rerank would not move these** — this is exactly the evidence DESIGN.md §7
  gates the graph-retrieval experiment on, and it points *away* from building it.
  The dominant FRAMES lever is a reasoning/compute step, not retrieval.
- **MuSiQue misses are extraction-dominated (60%).** The gold fact *was* in a chunk
  the worker saw, but no worker claim captured it — points at the worker prompt /
  model tier, not retrieval. Synthesis is a real secondary lever (30%: worker
  captured the fact, the report dropped it).
- **`retrieval_or_composition` deliberately conflates two things** (true retrieval
  miss vs. verbatim-absent computed answer); the per-question JSON keeps
  `gold_in_chunks`/`tag` so a spot-check separates them. `gold_contained` is a
  normalized-substring proxy for correctness, preferring stored judge scores when
  present.
- **This is single-run n=20 per benchmark** — enough to rank failure *modes* (the
  splits are lopsided, not marginal), not to quantify a lever's effect; that needs
  the repeat-3x discipline this doc applies elsewhere, applied per-lever next.

## Reproducing

```bash
python -m eval.run_eval --mode smoke --database-url sqlite+aiosqlite:///./phase2_failure.db
python scripts/failure_analysis.py --database-url sqlite+aiosqlite:///./phase2_failure.db
# --no-rerun-retrieval skips the ~30s/miss CPU rerank (leaves misses unsplit); --limit N caps questions
```

**Next (lever selection, not yet done — the data ranks these):** worker-extraction
lever (prompt + model-tier, MuSiQue-first) and a FRAMES numerical-reasoning
compute/verify step are the two the data supports; graph retrieval is *not*
indicated. Scope/budget for the lever ablations is a separate decision.

# Phase 2, lever 1 — worker extraction prompt (measured a wash, reverted) (2026-07-18)

First lever tried against the MuSiQue extraction misses: give the worker the overall
research question (it currently sees only its narrow sub-question) and a `worker_v2`
prompt telling it to extract salient cross-hop facts instead of returning empty
`open_gaps`. Rationale came from reading the actual failure traces (e.g. the "member
of Ratata" DOB worker had "11 September 1962" in a retrieved chunk but returned
`open_gaps` because it didn't know the date was relevant to the end goal).

**Result (n=20 MuSiQue, seed 42, vs the 0.50/0.415 baseline): a wash.**
`answer_contains_gold` 0.50→0.50, `answer_f1_extracted` 0.415→0.432 (within noise), at
**+20% tokens/cost**. The case inspection showed why: it fixed failures where the
bridging fact sat in a retrieved chunk (Adolf Overweg → now extracts "Chad Basin"),
but couldn't fix the ones where a *dependent* sub-question retrieves the wrong chunks
because the entity isn't resolved yet (the Ratata "DOB of each member" worker never
fetches Mauro Scocco's page — it doesn't know to search for him). **Reverted** (no
aggregate gain shouldn't cost +20%). The real finding: the dominant bottleneck is
**multi-hop dependency chaining**, which a worker prompt can't fix — it's a
planning/orchestration problem. That motivated lever 2.

# Phase 2, lever 2 — multi-hop self-corrective ReAct agent (`agent_mode=react_agent`) (2026-07-18)

Built a tool-calling agentic-RAG ReAct agent (`agent/react_agent.py`, DESIGN.md row 2):
LangGraph `StateGraph` with the prebuilt **`ToolNode`** executing a single `search`
tool, a ReAct loop where the model refines each query with entities resolved from
prior results (sequential multi-hop — the exact thing plan-first's parallel
decomposition can't do), a `finalize` node producing the structured cited report, and
a **self-correcting `verify` node** (reflection: is every hop answered & grounded?)
that loops back to search more, bounded by `max_corrections` + the budget ceilings.
The agent's tool-calling node uses a LangChain `ChatOpenAI` at OpenRouter (ToolNode
needs LangChain-format tool_calls, which the httpx `LLMClient` doesn't emit); its
`usage_metadata` is mapped through the same `_cost` table so accounting is consistent.
finalize/verify keep the `LLMClient`. Retrieval is the *same* `retrieve.retrieve_chunks`
the plan-first worker uses, so the ablation isn't confounded by two retrieval paths.

**Behavior check first** (the Ratata 2-hop that plan-first fails): the agent searched
"members of Ratata" → resolved Mauro Scocco → reported his birth date correctly, cited,
`verify` judged it sufficient, $0.0024. (Scored 0 on `answer_contains_gold` only because
of date *formatting* — gold "11 September 1962" vs the agent's "September 11, 1962",
identical date; the judge-extracted F1 metric credits it, the blunt substring check
doesn't — a metric nuance that if anything *under*counts react_agent.)

## Results (n=20 MuSiQue, seed 42, real Gemini 2.5 Flash) — `results/eval_custom_20260718T*.json`, `phase2_react_agent.db`

| Metric | plan_first (baseline) | react_agent | Δ |
|---|---|---|---|
| `answer_contains_gold` | 0.50 | **0.60** | **+0.10** |
| `answer_f1_extracted` | 0.415 | **0.457** | +0.042 |
| `answer_f1` (raw) | 0.069 | 0.056 | −0.013 |
| task_completion_rate | 1.00 | 1.00 | = |
| mean tokens/task | 9,035 | 20,391 | +126% |
| agent cost (20q) | $0.112 | $0.205 | +83% |
| wall-clock (20q) | 656s | 613s | −7% |

**Per-question flip analysis (same 20 questions):** react_agent 12/20 correct vs
plan-first 10/20. It **fixed 4** dependency-chained misses — two of them the hardest
**4-hop** questions, plus two 2-hop — and **regressed 2** (a 4-hop and a 3-hop where
plan-first's breadth won). Net +2, and the fixes land squarely on the multi-hop class
the design targets — mechanism-confirmed, not an aggregate coincidence.

## Reading this honestly

- **This is a real, mechanism-confirmed accuracy gain on multi-hop, at a real cost
  premium.** +10pts `contains_gold` / +4.2pts extracted-F1, fixing the hard 4-hop
  dependency chains, for ~2x tokens and +83% cost (multi-hop search + finalize +
  verify loops). Wall-clock is actually slightly *better* (no parallel-worker rerank
  contention). That's a genuine accuracy/cost tradeoff, not a free win.
- **Single-run n=20 is not yet conclusive.** This repo's own repeat-3x ablation
  measured ±12.6pt run-to-run variance on `contains_gold` at n=20 — a +10pt single-run
  gain sits at the edge of that. The per-question analysis (fixing 4-hop chains
  specifically) is stronger evidence than the aggregate alone, but the disciplined
  confirmation is a **repeat-3x** run before any default change (DESIGN.md row 2's
  stated bar). **Not run yet** — flagged as the next step, gated on spend.
- **The default stays `plan_first`.** Row 2's bar is accuracy parity *at lower/comparable
  cost*; react_agent wins accuracy at higher cost, so it ships as an additive,
  measured alternative, not the new default — pending the repeat-3x + a cost/accuracy
  call.
- **What this doesn't cover**: no FRAMES react_agent run yet (FRAMES misses are
  numerical composition, a different lever); no repeat-3x; `scripts/failure_analysis.py`
  can't yet classify react_agent runs (it recovers sub-questions from `worker`
  trajectories, which the agent doesn't produce — it uses `agent_step` + search
  tool_calls; a small tool enhancement, not done).

## Reproducing

```bash
DEEPRESEARCH_AGENT_MODE=react_agent python -m eval.run_eval --benchmark musique --n 20 --seed 42 \
  --database-url sqlite+aiosqlite:///./phase2_react_agent.db
```

# Phase 2, lever 2b — a `calculate` tool for FRAMES numerical reasoning (measured a wash) (2026-07-18)

The other open failure mode was FRAMES: 82% of misses are *computed* answers no source
states verbatim ("14 years younger", "43,518"), which a search-only agent structurally
can't produce. Added a second tool to react_agent's `ToolNode` — `calculate(expression)`,
a safe AST-whitelist arithmetic/comparison evaluator (never `eval()`; +-*/// % **,
comparisons, abs/round/min/max; injection-refusal unit-tested). The agent searches for
the operand facts, then calls `calculate` for the exact result. General capability, no
benchmark conditioning; both tools available on every run.

**The tool works and is genuinely used.** 7 calls across 6 of the 20 FRAMES questions,
on real numerical reasoning: `1847 - 1812 = 35`, day-counting `11 + 31 + 30 + 31 + 31 +
2 = 136`, `2.5 * 60 = 150`, `round((72 + 13 + 1) / 10) * 10 = 90`. One attempt failed —
`days_between('1945-04-19', '1945-09-02')` — the evaluator has no date arithmetic (a
real, un-closed gap for FRAMES date-difference questions).

## Results (n=20 FRAMES, seed 42, real Gemini 2.5 Flash) — `phase2_react_frames.db`

| Metric | plan_first (baseline) | react_agent + calculate | Δ |
|---|---|---|---|
| `accuracy` (LLM-judged) | 0.45 | 0.45 | **0.00** |
| `citation_coverage` | 0.702 | 0.581 | −0.121 |
| `citation_precision` | 0.758 | 0.548 | −0.210 |
| mean tokens/task | 10,714 | 34,104 | +218% |
| agent cost (20q) | $0.117 | $0.276 | +136% |

**Per-question flip:** both 9/20 correct. `calculate` **fixed 4** numerical questions,
but react_agent **regressed 4** others — a wash.

## Reading this honestly

- **The `calculate` tool did its job; react_agent as a whole did not help FRAMES.** The
  tool fixed the 4 questions that were pure arithmetic-over-gathered-facts. But
  react_agent's *sequential* search regressed 4 others, and citations dropped sharply —
  because **FRAMES rewards retrieval breadth** (2–15 articles per question, gather many
  independent facts), which plan-first's *parallel* worker pool does well and a
  one-query-at-a-time agent sacrifices. Net accuracy unchanged, at 2.4x cost and worse
  grounding.
- **The load-bearing insight: topology should match question shape.** react_agent
  (sequential depth + self-correction) wins MuSiQue's *dependency chains* (+10pts,
  above); plan-first (parallel breadth) wins FRAMES's *wide multi-doc* questions. This
  is why the design keeps plan-first the default and ships react_agent as an additive,
  config-selectable mode rather than a replacement — the right agent depends on whether
  the workload is depth-heavy or breadth-heavy. A future router could pick per-question;
  not built (no metric yet says the routing itself is worth its complexity).
- **`calculate` stays** (it's correct, cheap when unused, and the FRAMES-numerical fix
  is real) — but it doesn't move the FRAMES aggregate on its own, and the date-arithmetic
  gap is left open, documented rather than hidden.

## Reproducing

```bash
DEEPRESEARCH_AGENT_MODE=react_agent python -m eval.run_eval --benchmark frames --n 20 --seed 42 \
  --database-url sqlite+aiosqlite:///./phase2_react_frames.db
```

## 2026-07-18 — unified planner→supervisor→subagent→verify→synthesis→reflection rebuild

Direct response to this doc's own "topology should match question shape"
finding immediately above: rather than ship plan-first (parallel breadth) and
react_agent (sequential depth) as two config-selectable modes and leave the
router-vs-fixed-default question unresolved, this rebuild collapses them into
**one** graph where the *supervisor* — not a global mode flag — decides
parallel-vs-sequential dispatch per plan node, based on the node's actual
`depends_on` edges. Independent facets (FRAMES-shaped breadth) still fan out
in parallel; dependent chains (MuSiQue-shaped depth) still resolve hop by hop
with entity substitution; both live in the same DAG.

**What this session actually measured: none of it against a real model.**
Every claim below is offline-only — stubbed structured-JSON LLM responses and
a stubbed LangChain chat model, no network, no `ANTHROPIC_API_KEY` /
`OPENROUTER_API_KEY` spent. This is a harness-correctness/mechanics
verification pass (does the graph wire together, does the checkpointer
actually resume, do the readiness/verify/reflection gates fire when they
should), the same caveat this doc has flagged on every `FakeLLMClient`-era
entry above — read it the same way: real control flow, zero quality signal.

**Full test suite: 94/94 green** (`tests/test_planner.py`,
`test_subagent.py`, `test_graph.py`, `test_checkpointer.py`,
`test_orchestrator_persistence.py`, plus the pre-existing suite untouched by
this rebuild). Specific mechanics proven, not merely asserted-in-code:

- **Wave sequencing**: a 3-node DAG (2 independent lookups + 1 node
  depending on both) dispatches the two independent nodes together and the
  dependent node only after both upstreams are verified, with the
  dependent node's brief provably containing both upstream answers
  (`facts_for()` injection).
- **Per-hop verify gate**: a node feeding a later hop is gated by a paid
  grounding check; a leaf node (nothing depends on it) is marked verified
  for free — a forced verify-fail requeues the node exactly once before
  succeeding; a verify that never passes gives up cleanly at the
  `max_corrections` boundary and the run still reaches synthesis rather
  than hanging.
- **Dependency-ordered synthesis + bounded reflection**: findings are fed
  to synthesis in topological order regardless of the order waves actually
  completed in; a detected report gap spawns exactly one bounded follow-up
  wave then confirms clean; a reflection that never converges is capped by
  `max_reflect`, not an unbounded loop.
- **Checkpointer**: a run interrupted right after `plan` (via LangGraph's
  own `interrupt_after`, not a manually-raced async generator) resumes from
  that exact checkpoint in a **fresh graph object** — proven by a
  call-counting stub showing `plan_node` never re-executes on resume — and
  a different `thread_id` against the same checkpointer file starts a
  genuinely independent run.

**Two real bugs caught by this session's own tests, before any live-model
run** (both introduced by this rebuild's new code, not pre-existing —
neither surfaced until a test exercised a genuine multi-turn tool-calling
loop with a live budget check):

1. The supervisor's `Send` payload to each subagent never carried
   `started_monotonic` forward, so every subagent's inner ReAct loop
   computed elapsed wall-clock time from epoch zero on its first budget
   check — instantly (and silently) tripping the wall-clock ceiling and
   skipping every tool call after exactly one turn. A test asserting
   `search`/`fetch` tool_calls actually got recorded (ported from the old
   worker-based persistence test) caught this: the finding came back with
   `"(no passages retrieved)"` even though the stub chat model had
   requested a search.
2. The `search`/`calculate` tools recorded `tool_calls.span_id` from a
   static value captured once when the run's context was built, instead of
   the OTel span actually active at call time — breaking the invariant
   (already covered by an existing test) that every `tool_calls` row's
   `span_id` names a real `trajectories` row it nests under.

Both are now fixed and covered by regression assertions in the tests above.

**Not yet done — the actual accuracy question**: a same-day follow-up (next
entry below) ran one real 2-hop question live and it worked correctly
end-to-end — a genuine smoke-level signal that the rebuild functions against
a real model, not just stubs. But that is one hand-picked question, not a
benchmark. Still open: does the unified graph capture react_agent's measured
MuSiQue multi-hop win (+10pts `answer_contains_gold`, above) without paying
its ~2x-token/+83%-cost premium on FRAMES-shaped independent-facet
questions? That requires a live-model run (MuSiQue + FRAMES smoke subsets,
scored per-hop) — not performed this session, and the honest gate before
citing this rebuild as a quality improvement rather than a harness change.

**Reproducing** (offline, no key required):

```bash
python -m pytest tests/test_planner.py tests/test_subagent.py tests/test_graph.py \
  tests/test_checkpointer.py tests/test_orchestrator_persistence.py -q
```

## 2026-07-18 (same day, follow-up) — Instructor for structured output; first live-model verification

Two follow-ups to the rebuild above, done in the same session. First: a live
smoke test against a real model (`google/gemini-2.5-flash` via OpenRouter,
the key already configured in `.env`), the first real-model run of the
unified graph — everything above this entry had been offline/stubbed only.
Second: `llm/client.py`'s `complete_json` was rewritten around the
`instructor` library (`response_model=<PydanticModel>` instead of a
hand-maintained JSON-schema dict + manual `json.loads`), renamed
`complete_structured`, with the same behavior-preservation discipline as the
graph rebuild — verify before trusting, don't assume a library's default is
correct.

**First live run, before the Instructor swap** (still the old hand-rolled
`output_config`/`response_format: json_schema` path): a 2-hop dependency
question — "What year was the lead singer of the Swedish pop group Ratata
born in?" against a 2-document `LocalCorpusBackend` — completed correctly
end-to-end. Real planner produced the correct DAG (`n1`: find the singer,
independent; `n2`: find the birth year, `depends_on: ["n1"]`); `facts_for()`
correctly injected "Mauro Scocco" into n2's brief; both hops verified
(confidence 1.0); reflection ran once and correctly found no gaps; synthesis
produced a correctly-cited one-sentence answer. Cost: $0.0038, 6,942 tokens,
`COMPLETED`. This is the first evidence (beyond offline test mechanics) that
the rebuilt graph actually works against a real model on the exact scenario
it was built for.

**A real bug the Instructor swap surfaced, caught before it shipped**:
Instructor defaults to `Mode.TOOLS` (forcing a synthetic tool call shaped
like the Pydantic model) for both its Anthropic and OpenAI-compatible
clients. Live-tested directly against this project's actual nested
`Plan{sub_questions: list[SubQuestion]}` schema on `google/gemini-2.5-flash`:
`Mode.TOOLS` (and `Mode.TOOLS_STRICT`) consistently **flattened every
`SubQuestion` into a plain string** instead of a nested object — across 4
retries (Instructor's automatic validation-retry, which re-prompts the model
with the validation error), the model never converged; one retry attempt
even tried jamming a JSON-encoded string *inside* the string slot, getting
worse, not better. Switching to `Mode.JSON_SCHEMA` (the API's native
structured-output/constrained-decoding feature — the same mechanism the
pre-Instructor code already used natively) fixed it immediately, first try.
**Lesson recorded for future work with Instructor**: never leave the mode at
its library default for a nested-object schema without testing against the
actual model in use — a smaller/cheaper model's function-calling schema
generation can silently fail to preserve nesting, and the failure mode
(flattened strings, retried into worse attempts) doesn't look like an
obvious "this library is broken" signal, it looks like the model being
confused, so it's easy to misattribute.

**Second live run, after the fix** (`Mode.JSON_SCHEMA` for OpenRouter,
`Mode.ANTHROPIC_JSON` chosen by the same reasoning for the Anthropic/Bedrock
path but **not live-verified this session** — Anthropic credit still
exhausted, per the caveat above): identical question, same corpus. Same
correct DAG, same correct dependency injection (`entities_extracted:
{'lead_singer': 'Mauro Scocco', 'group': 'Ratata'}`), same verified findings,
same correctly-cited report. Cost: $0.00379, 6,952 tokens — within
rounding of the pre-Instructor run. **Net: Instructor is a drop-in
replacement at parity cost, once the mode is pinned correctly** — the
hand-maintained JSON-schema dicts are gone (Pydantic models are now the
single source of truth for every structured LLM call in the codebase:
`Plan`, `FindingDraft`, `HopVerdict`, `GapCheck`, `SynthesisDraft`,
`AccuracyVerdict`, `CitationVerdict`, `ShortAnswer`, `RaceScores`), and every
call now gets automatic validation-retry for free.

**Reproducing** (live, needs `OPENROUTER_API_KEY` in `.env`):

```bash
python -c "
import asyncio
from deepresearch.agent.orchestrator import run_research
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.telemetry.otel_setup import init_telemetry

DOCS = [
    {'doc_id': 'd1', 'title': 'Ratata', 'text': 'Ratata is a Swedish pop group fronted by Mauro Scocco.'},
    {'doc_id': 'd2', 'title': 'Mauro Scocco', 'text': 'Mauro Scocco was born on 11 September 1962 in Stockholm, Sweden.'},
]

async def main():
    init_telemetry()
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    result = await run_research(
        'What year was the lead singer of the Swedish pop group Ratata born in?',
        config=config, search_backend=backend,
    )
    print(result.status, result.report.text if result.report else None)

asyncio.run(main())
"
```

