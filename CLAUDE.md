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
  explicit and configurable: per-hop retries (`max_corrections`), bounded
  post-synthesis reflection (`max_reflect`), planner over-decomposition
  ceiling (`max_nodes`), budget ceiling.
- Redis is a cache (search-query results + fetched pages, keyed + TTL'd) — **not**
  agent memory. Never repurpose it for cross-run reasoning state.
- Report distributions, never point estimates. Reliability evals repeat a subset
  3–5× and report variance / consistency, not a single score.
- Config-next-to-result: every stored eval score and CI baseline carries the exact
  config JSON + git SHA that produced it.

## Key design decisions (see docs/DESIGN.md §2 for full table + reversal evidence)

- Topology: one LangGraph StateGraph — planner emits a dependency DAG, a
  supervisor router fans out ready nodes wave-by-wave (parallel where
  independent, sequential where dependent), each node is a tool-calling
  ReAct subagent with an inline per-hop verify gate (§11 in DESIGN.md).
- Planning: single planner + subagent design (no more plan-first-vs-react_agent
  mode flag); bounded per-hop retries (`max_corrections`) and bounded
  post-synthesis reflection (`max_reflect`).
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

Tuned against a real measured baseline and real measured single-run variance
(`docs/RESULTS.md`), not the original placeholders:

- PR smoke gate (`scripts/ci_gate.py`) actually *fails* the check only on
  structurally low-variance metrics: `cost_per_query_usd` (+25% relative) and
  `task_completion_rate` (-3pts absolute). `accuracy` / `citation_precision` /
  `answer_f1*` are computed and shown in the PR comment every run, but do not
  fail the check — the repeat-3x architecture ablation measured single-run
  noise up to ~17-25 points on this exact metric family at n=20, past any
  flat tolerance that could both survive that noise and still catch a real
  regression of similar size (see `eval/ci_baseline.py`'s
  `INFORMATIONAL_ONLY_METRICS`). Real regressions on those metrics are meant
  to be caught by the nightly variance-aware policy below instead, once it's
  built against a real distribution rather than a single point.
- Nightly full suite: no auto-gate, but any metric outside ±1 stdev of the last
  5 nightly runs should be flagged in the artifact for manual review — not yet
  implemented (nightly currently only refreshes the baseline on green); a
  flagged follow-up now that PR-smoke runs a real model.
- Reliability: a 20-question subset must show an all-runs-consistent (pass^k) rate
  reported alongside every accuracy number — an accuracy figure without it is
  incomplete and should not be cited on its own.
- DeepResearch Bench: weekly 10-question EN subset only; full 100-task suite is
  monthly/manual (judge + execution cost makes nightly/weekly-full unaffordable —
  see docs/DESIGN.md §5.6).
