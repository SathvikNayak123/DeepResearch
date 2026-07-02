# DeepResearch

Agentic deep-research system whose real product is the eval harness and AIOps
around it — every architectural box ships with a metric attached, or it doesn't
ship.

**Read `docs/DESIGN.md` first.** It is the source of truth for architecture, the
decision table (with alternatives and reversal evidence for every choice), the
eval design, run-store schema, and the session map. Don't re-derive a decision
that's already in that table — extend it if new evidence arrives.

## Universal rules

- Every architectural box must have a metric attached, or it doesn't ship.
- "The agent decides" is not a stopping criterion. Stopping logic is always
  explicit and configurable: max iterations/replans, coverage self-check
  threshold, budget ceiling.
- Redis is a cache (search-query results + fetched pages, keyed + TTL'd) — **not**
  agent memory. Never repurpose it for cross-run reasoning state.
- Report distributions, never point estimates. Reliability evals repeat a subset
  3–5× and report variance / consistency, not a single score.
- Config-next-to-result: every stored eval score and CI baseline carries the exact
  config JSON + git SHA that produced it.

## Key design decisions (see docs/DESIGN.md §2 for full table + reversal evidence)

- Topology: orchestrator + bounded parallel sub-question worker pool (max 4–6).
- Planning: plan-first with bounded re-planning (≤2 replans on reflection).
- Context: structured notes (claim/source_id/quote/confidence) between workers and
  synthesis, never full-transcript stuffing. Token budgets enforced per stage.
- Search: Tavily primary, behind a `SearchBackend` protocol with a
  `LocalCorpusBackend` for fixed-corpus benchmark/CI runs.
- Citation: structured claim→source_id mapping, checked post-hoc (FACT-style) for
  coverage + precision.
- Retrieval: self-hosted `bge-reranker-v2-m3` default, Cohere optional, same
  interface. With/without-rerank ablation from day one.
- Run store: Postgres (`runs`, `trajectories`, `tool_calls`, `eval_scores`,
  `ci_baselines`). `run_id` doubles as the OTel trace ID.
- Observability: Langfuse (OTel-native) + Prometheus/Grafana for infra metrics.
- Non-goals: no graph retrieval in v1 (deferred, hypothesis stated), no
  cross-session memory (no benchmark measures it), no multi-tenant auth.

## Eval thresholds

Initial gating thresholds — placeholders until the first real baseline lands in
`ci_baselines`, then tune against measured variance rather than guessing:

- PR smoke gate fails if, vs. stored baseline: FRAMES-20 or MuSiQue-20 accuracy
  drops >5 points absolute, citation precision drops >5 points, or p95 latency
  regresses >30%.
- Nightly full suite: no auto-gate, but any metric outside ±1 stdev of the last
  5 nightly runs gets flagged in the artifact for manual review.
- Reliability: a 20-question subset must show an all-runs-consistent (pass^k) rate
  reported alongside every accuracy number — an accuracy figure without it is
  incomplete and should not be cited on its own.
- DeepResearch Bench: weekly 10-question EN subset only; full 100-task suite is
  monthly/manual (judge + execution cost makes nightly/weekly-full unaffordable —
  see docs/DESIGN.md §5.6).
