"""Citation coverage/precision — docs/DESIGN.md decision row 6, FACT-style:
coverage = % of gathered claims that made it into the cited report;
precision = % of those cited claims a judge confirms are entailed by the
quote they cite."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.schemas import RunResult

from eval.judge import Judge


@dataclass
class CitationMetrics:
    coverage: float
    precision: float
    n_claims_checked: int

    def summary(self) -> dict:
        return {"coverage": self.coverage, "precision": self.precision, "n_claims_checked": self.n_claims_checked}


async def compute_citation_metrics(result: RunResult, judge: Judge) -> CitationMetrics:
    if result.report is None:
        return CitationMetrics(coverage=0.0, precision=0.0, n_claims_checked=0)

    # Which source_ids the model actually cited -- read directly from the
    # structured synthesis output (Report.citations, populated from
    # SynthesisDraft.cited_source_ids), not re-derived by regex-parsing the
    # free-form report text. A regex-based version of this broke silently on
    # real model output that cited multiple ids in one bracket
    # (`[src_a, src_b]` -- `\w+` can't span the comma, so the whole marker
    # was missed); reading the already-validated structured field sidesteps
    # every such formatting variance entirely.
    cited_source_ids = {c.source_id for c in result.report.citations}

    all_claims = [claim for notes in result.worker_notes for claim in notes.claims]
    checked_claims = [claim for claim in all_claims if claim.source_id in cited_source_ids]
    coverage = len(checked_claims) / len(all_claims) if all_claims else 0.0

    if not checked_claims:
        return CitationMetrics(coverage=coverage, precision=0.0, n_claims_checked=0)

    supported = 0
    for claim in checked_claims:
        is_supported, _, _ = await judge.judge_citation(claim=claim.text, quote=claim.quote)
        if is_supported:
            supported += 1
    precision = supported / len(checked_claims)

    return CitationMetrics(coverage=coverage, precision=precision, n_claims_checked=len(checked_claims))
