# DeepResearch — Adversarial Audit Report

**Auditor role:** independent technical auditor trying to break the "it's complete" claim.
**Method:** verified by running the test suite, recomputing baselines from the SQLite run
store, grepping git history across all branches, and reading the Terraform / CI / CD source
directly — not by trusting README, DESIGN.md, RESULTS.md, comments, or the prior audit report.
**Date:** 2026-07-05 · **Branch audited:** `bootstrap-ci-and-ablation` (HEAD `f2a0a66`),
with an explicit contrast against the default branch `origin/main`.

> This supersedes the earlier audit that ran against HEAD `7b52e9b`. Two of that report's
> three headline blockers have genuinely moved: **`infra/` is now committed** (commit `f2a0a66`,
> 34 files + a real 1007-line `terraform plan`), and **real-model agentic evidence now exists**
> (18 completed real-key runs, $2.24 spent, in `real_baseline.db`). But the completeness claim
> still breaks — the failure mode has shifted from "infra isn't committed" to **"the default
> branch is empty and every gap-closing artifact is uncommitted working-tree state."**

---

## Verdict: **FIX-FIRST**

The engineering *substance* is real and unusually honest. The test suite is green (52 passed),
every published number recomputes exactly from raw run-store rows, the IAM design is genuinely
least-privilege, the OIDC trust policy is correctly repo/branch-scoped, and the run store now
contains real-model trajectories that actually decompose and re-plan. I could not find a single
number that misrepresents what produced it; the fake-client / no-key / credit-exhausted caveats
are stated loudly and in the right place. That remains this repo's strongest quality.

It is **not SHIP-ready**, for reasons that are now overwhelmingly about **git/commit state**
rather than missing work — which is both good news (the work largely exists) and a real blocker
(a reviewer cannot see it in a clean clone):

1. **The default branch is effectively empty.** `git clone` checks out `origin/main`, which
   contains **only `.gitignore`, `LICENSE`, and `README.md`** — no `src/`, `eval/`, `infra/`,
   `docs/`, tests, or workflows. The README's very first instruction ("Read `docs/DESIGN.md` and
   `CLAUDE.md` first") 404s on a fresh clone. The entire project lives on
   `bootstrap-ci-and-ablation`, 8 commits ahead of `main` and never merged.
2. **Every artifact that closes the prior audit's gaps is uncommitted.** The CD workflow
   (`deploy.yml`), the GitHub-OIDC Terraform module, the real-model agentic proof
   (`docs/proof/real_agentic_trajectories.json`), the DESIGN.md deploy decision rows 12–13, the
   RESULTS.md dogfood section, the API-key auth, and the Haiku effort-param fix are **all
   untracked / unstaged working-tree changes**. They are good work sitting outside version control.
3. **No deploy ever ran.** The CD pipeline's auto-rollback is coded but never exercised; the only
   Actions-run links in the repo are the three CI-gate PRs (runs 7/8/9). The "one hands-free
   deploy + one failed-smoke auto-rollback" evidence requirement is unmet.
4. **Real research quality is still unpublished.** The real-key run ran out of Anthropic credit at
   18 of ~20 FRAMES questions, and no aggregate real accuracy/F1 was scored or committed — so the
   headline "what is the actual FRAMES accuracy?" still has no answer in the repo.

Fixes #1 and #2 are minutes of `git` work (merge the branch, `git add` the working tree) and are
the highest-leverage changes. #3 and #4 are real scope (one real deploy run; one funded eval run).

---

## PASS / GAP table

### Universal checks

| # | Check | Status | Evidence / fix |
|---|---|---|---|
| U1 | Clean-clone documented run | **GAP (severe)** | `origin/main` = `{.gitignore, LICENSE, README.md}` only (`git ls-tree origin/main`); 0 files under `src/`, `eval/`, `infra/`, `docs/`. A `git clone` yields a README pointing at `docs/DESIGN.md`/`CLAUDE.md` that **do not exist on that branch**. Everything real is on `bootstrap-ci-and-ablation` (8 commits ahead, unmerged). Even on that branch the headline `python -m deepresearch.cli "…"` calls the real Anthropic+Tavily APIs with no fake fallback ([cli.py](../src/deepresearch/cli.py), [config.py](../src/deepresearch/config.py)), so a keyless clone fails; the offline path is `python -m eval.run_eval --mode smoke` plus `DEEPRESEARCH_RERANK_ENABLED=false` (else a silent ~1 GB `bge-reranker-v2-m3` download). The README now documents both caveats — but that edit is **uncommitted**. **Fix:** merge `bootstrap-ci-and-ablation` → `main`; commit the README edits. |
| U2 | Claim→artifact tracing | **PASS (with caveat)** | Recomputed `musique.answer_f1` from raw `eval_scores` = **0.0210393583192** over n=120 vs. committed `results/ci_baseline.json` 0.0210393583175 — match to ~10 sig figs; `frames.accuracy = 0.700` over n=20 exact. Caveat: all committed accuracy/F1/cost figures are `FakeLLMClient` output (`total_cost_usd = 0` on all 120+ committed runs) — disclosed loudly. |
| U3 | Reproduce a headline number | **PASS** | 52/52 tests pass in 3.2 s; baseline recompute above reproduces the published figure from the run store; the prior cache-measurement cold/warm/mixed/bypass shape reproduced offline. |
| U4 | CI gate fails on regression | **PASS (narrow)** | `scripts/ci_gate.py` exits 1 and names the metric; workflow re-raises ([pr-smoke.yml](../.github/workflows/pr-smoke.yml)); real red run exists (PR #3 / Actions run 9). **But** by the harness's own CI design (FakeLLMClient, no key) 3 of 4 gated metrics are structurally immovable, so the gate rests on `task_completion_rate`, and the demonstrated regression is a *total collapse* (`max_total_tokens=10` → every run `budget_exceeded`), not a subtle drop. Honestly documented in RESULTS.md ("A real gap this session's own sanity-checking surfaced"). |
| U5 | Design-doc ↔ code integrity | **GAP (drift)** | (a) MuSiQue **Support F1** promised (DESIGN §5.1) but never computed — `support_f1` appears only as a comment string in [store/models.py:104](../src/deepresearch/store/models.py#L104), in no metric. (b) Session-6 **SSE streaming** + **`GET /runs/{id}`** (`api/streaming.py`, `routes_runs.py`) do not exist — `src/deepresearch/api/` is `{__init__, main}.py`; the only working-tree change to `main.py` is an API-key `Depends`, not streaming. (c) Decision-table **rows 12–13** (the ECS/Fargate & data-layer justification the whole deploy story rests on) exist **only in the uncommitted working tree** — HEAD's DESIGN.md has no rows 12–13. |
| U6 | Eval economics | **PASS** | Cheap judge default `claude-haiku-4-5`; verdicts cached in `judge_cache` keyed on `sha256(rubric+example+answer)`; estimate printed before each run; smoke/nightly/DRB split real. PR cycle ≈ $0 and nightly ≈ $0 because CI runs FakeLLMClient — cheap, but CI therefore never exercises real cost. |
| U7 | Secrets / hygiene | **PASS** | No `sk-ant-`/`tvly-` strings in history; `.env`, `*.tfstate`, `*.db` gitignored and uncommitted (the real-key stores `real_baseline.db`/`deepresearch_dogfood.db` are correctly kept out). Deps pinned `~=`; MIT LICENSE; dataset licenses cited (MuSiQue CC BY 4.0, DRB Apache-2.0). |
| U8 | README voice | **PASS (with a stretch)** | Notably free of hype; heavy, correctly-placed caveats. Stretch: README L8 "all wired end-to-end" is strong when the deploy layer isn't on the default branch and the CD/proof are uncommitted. |

### Project checks — Deep Research Agent

| # | Check | Status | Evidence / fix |
|---|---|---|---|
| P1 | Agentic, not pipeline | **PASS (evidence not yet committed)** | Now genuinely demonstrated with a **real model**: `real_baseline.db` holds 18 completed real-key FRAMES runs, workers-per-run distributed **2→8** (max 8), **49 reflections**, real re-planning. Exhibit run `12324cd1…` in [docs/proof/real_agentic_trajectories.json](proof/real_agentic_trajectories.json): a 4-sub-question plan → 8 workers → 3 reflections firing **2 real replans** (`should_replan: true`, `coverage_score` 0.5→0.4→0.7), then synthesis. This decisively answers the prior audit's "every committed run is 1 worker / 0 replans." **Gap:** the proof JSON is **untracked** and `real_baseline.db` is gitignored — a clean clone still sees only the FakeLLM (1-worker) runs. **Fix:** `git add docs/proof/real_agentic_trajectories.json` (it's already written; it just isn't staged). |
| P2 | Benchmark protocol fidelity | **PASS (one gap)** | MuSiQue Answer F1 is correct SQuAD-style token-overlap, best-of-aliases ([answer_f1.py](../eval/metrics/answer_f1.py)); revisions pinned; stratified sampling real. Citation precision is entailment-judged against the **actually-fetched source quote**, not string overlap ([citation.py](../eval/metrics/citation.py)) — FACT-style, correct. **Gap:** MuSiQue **Support F1** (a core official metric) absent (see U5a). FRAMES uses the cheap judge, not the paper's rater (disclosed). |
| P3 | Reliability reporting | **PASS** | `run_reliability` does 20×3, reports per-repeat list, mean, stdev, and pass^k all-consistent rate ([run_eval.py](../eval/run_eval.py)); per-repeat verdict via `gold_contained`, **not** best-of. No README number is a best-of run. Values are FakeLLM (disclosed). |
| P4 | Ablation integrity | **PASS** | Rerank ablation uses the real `CrossEncoderRerankBackend`, full config committed; cache ablation configs committed; plan-first-vs-ReAct addendum (DESIGN §10) is honest — states the accuracy axis is *untestable* under FakeLLM and neither decision-table row is reversed, and marks its `git_sha` as `no-git`. |
| P5 | Budget enforcement | **PASS** | `tests/test_budget.py` (tiny `max_total_tokens`/`max_usd`/`max_wall_clock`) is in the 52-green suite; budget is checked at stage boundaries; PR #3 (`max_total_tokens=10`) drove every run to `budget_exceeded` — graceful stop, logged ceiling, no crash. |
| P6 | Cache honesty | **PASS** | Bypass = the `CachedSearchBackend` wrapper is never constructed ([backends/cached.py](../src/deepresearch/backends/cached.py)) — proven by the bypass pass matching cold. Hit-rate from `CacheStats`; `$-saved = hits × per-call cost`. Caveats disclosed (`fakeredis` + `FakeTavilyBackend`; Grafana panels unverified vs live traffic). |
| P7 | Run store recompute | **PASS** | Independently recomputed the CI baseline from raw `eval_scores`/`runs` — matches `ci_baseline.json` (see U2). Schema matches DESIGN §4 + a documented `judge_cache` addition. |
| P8 | Infrastructure as code | **PASS on content / GAP on reachability** | `infra/` is now committed (`f2a0a66`, 34 files) with a real `terraform plan` proof (`docs/proof/plan.txt`, "Plan: 35 to add, 0 to change, 0 to destroy"). IAM is genuinely least-privilege: **separate execution vs task roles** ([modules/iam/main.tf](../infra/modules/iam/main.tf)); task role scoped to `ssm:Get*` on its own path, logs on its own group, and `cloudwatch:PutMetricData` condition-scoped to its namespace; the only `Resource:"*"` entries (`kms:Decrypt` via-SSM-condition, `PutMetricData`, ECS describe/register) are AWS-unavoidable and each condition-scoped or read-only. Secrets are SSM `SecureString`, not in the image/task-def. **Gap:** none of this is on `main`; `terraform plan` here is a from-scratch plan (no live stack to diff — the stack is torn down, `tfstate` empty), so "non-empty diff vs live = drift" is untestable, not clean-verified. |
| P9 | Keyless CD (OIDC) | **GAP (exists, uncommitted, never run)** | `deploy.yml` + `modules/github_oidc` now exist and are correct in shape: `id-token: write`, `aws-actions/configure-aws-credentials`, no stored keys; trust policy pins `aud=sts.amazonaws.com` **and** `sub=repo:{owner}/{repo}:ref:refs/heads/{branch}` — any other repo/branch/PR is denied ([github_oidc/main.tf](../infra/modules/github_oidc/main.tf)); auto-rollback to the prior task-def is coded on failed stabilize/health-check. **But:** (a) both are **uncommitted**; (b) `deploy.yml` triggers `on: push: [main]` — where **no application code lives**, so it would deploy an empty repo; (c) **no deploy run has ever executed** — the rollback path is unexercised and the required "one real hands-free deploy + one failed-smoke auto-rollback" evidence does not exist. (`scripts/deploy.sh` is the manual path — keyed local `ecr get-login-password`, no rollback — separate from the keyless CD.) |
| P10 | Deployment / teardown | **MIXED** | Teardown: `tfstate` empty → $0-forward; `residual_check.sh` is a thorough tag+name sweep with sound eventual-consistency cross-checks; $25/mo budget alarm codified with 80%/100% thresholds ([modules/budget/main.tf](../infra/modules/budget/main.tf)) plus a documented manual pre-apply alarm. ECS-vs-EKS decision row exists and matches reality (Fargate). **But:** live ALB URL unverifiable (stack down, infra off `main`); README gives cost *shapes* but **no single total-AWS-spend figure**; Fargate cold-start absent from any latency table (and every latency table is FakeLLM/simulated anyway). |
| P11 | Dogfood case study | **GAP (external + unresolvable from clone)** | The dogfood is *real* (verified real-key runs with non-zero cost in the untracked stores). **But** `CASE_STUDY.md` lives in a *separate* repo (`trace-replay`); the trace store it references is untracked; and it documents a **MuSiQue-run rerank crash**, not a *nightly* failure. Not resolvable from a clone of this repo. |

---

## The three weakest points an interviewer would attack

1. **"I cloned your repo and it's empty."**
   The default branch `main` is `README + LICENSE + .gitignore`. The README's first instruction
   references files that aren't there. Everything — code, evals, infra, the CD you just built —
   is stranded on an unmerged feature branch, and the artifacts that close your own audit's gaps
   (CD workflow, OIDC module, agentic proof, DESIGN rows 12–13) are uncommitted working-tree
   changes. A reviewer's first 30 seconds produce nothing runnable. This is a *packaging* failure
   sitting on top of genuinely good work, which makes it the most frustrating kind.

2. **"You say keyless CD with auto-rollback — show me a run."**
   The OIDC trust policy and the rollback logic are correct on paper (I read both). But no deploy
   workflow has ever executed: the only Actions runs in the repo are the three CI-gate PRs, the
   workflow is uncommitted, and it targets `main` where no app code exists. So the deploy story is
   *designed*, not *demonstrated* — the exact "vaporware until it runs once" critique the checklist
   is built to catch.

3. **"What is your agent's actual FRAMES accuracy?"**
   Real-model runs now exist and prove the loop genuinely decomposes and re-plans — a real
   improvement. But the real-key eval ran out of Anthropic credit at 18 of ~20 questions and **no
   aggregate accuracy/F1 was scored or committed**. Every committed headline number is still
   FakeLLMClient (cost $0). The honest framing is a strength; the missing answer is still a hole an
   interviewer will put a finger on.

---

## The single strongest evidence-backed interview claim this repo supports

> **"My eval scores are reproducible from raw rows and the regression gate fails closed — I
> recompute my published `musique.answer_f1` baseline of 0.0210393583 exactly from 120
> `eval_scores` rows in the run store, the CI gate exits non-zero on a real regression (PR #3), the
> budget ceiling stops a run gracefully, and the cache-bypass flag genuinely bypasses — all
> verifiable offline. And when I finally ran a real model, the run store shows the loop actually
> decomposing a 4-part plan into 8 workers across 2 reflection-triggered replans, not a
> retrieve-then-summarize pipeline."**

Every clause is independently verified: the baseline recompute matched to ~10 sig figs; 52/52
tests pass; the real trajectory in `docs/proof/real_agentic_trajectories.json` shows genuine
replanning at real (`$0.17`) cost. It is a claim about **evals-and-AIOps discipline** — the
project's stated real product — and it deliberately does **not** assert research-quality numbers,
which the repo cannot yet back.

---

## Highest-leverage remediation, in order

1. **`git merge bootstrap-ci-and-ablation` → `main`** (or open the PR and merge it). Fixes the
   clean-clone failure that dominates everything else.
2. **Stage and commit the working tree**: `deploy.yml`, `infra/modules/github_oidc/`,
   `docs/proof/real_agentic_trajectories.json`, and the DESIGN/RESULTS/README/`api`/`llm` edits.
   These are done — they just aren't in git.
3. **Run the deploy once** and capture (a) a green hands-free deploy and (b) a deliberately-broken
   image that fails the health check and auto-rolls-back — link both Actions runs, as the CI-gate
   PRs already do.
4. **Fund one `eval-smoke` FRAMES+MuSiQue run to completion** and commit the aggregate real
   accuracy/citation-precision as a second, real-model baseline alongside the FakeLLM one.

*Full checklist coverage, per-file references, and reproduction commands are inline above. The
dominant blocker is no longer missing work — it is that the work lives on an unmerged branch and in
an uncommitted working tree.*
