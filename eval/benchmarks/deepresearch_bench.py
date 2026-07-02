"""DeepResearch Bench — manual-only, gated, cost-printed-before-running.

docs/DESIGN.md decision row 11: judge-cost analysis did NOT approve this for
nightly (or even weekly-full) — RACE (GPT-5.5) + FACT (GPT-5.4-mini) judging
alone runs ~$15-35/100 questions, and adding this project's own agent
execution cost pushes a full-100 run toward ~$120-330. Per this session's
brief, that means: no automated subset here, only a manual `make eval-drb`
target that prints the documented cost estimate and requires explicit
confirmation before doing anything.

This is intentionally a stub past the cost-gate, not a full RACE/FACT
implementation — wiring the actual `Ayanami0730/deep_research_bench`
RACE/FACT judge pipeline (which judges full research *reports* against a
reference report, not short answers — a different shape of scoring than
FRAMES/MuSiQue's this session built) is real, separate work, explicitly
lower priority than FRAMES/MuSiQue in this session's task list. Treat this
as the guardrail + honest placeholder, not the finished benchmark.
"""

from __future__ import annotations

import argparse
import sys

# Figures from docs/DESIGN.md §5.6 / decision row 11 — not measured here,
# carried over from that doc's own researched estimate.
COST_ESTIMATES_USD = {
    "weekly": (15, 35),  # 10-question EN-only subset
    "full": (120, 330),  # 100-task suite (50 zh/50 en), monthly/manual only
}


def print_cost_estimate(mode: str) -> None:
    low, high = COST_ESTIMATES_USD[mode]
    n_questions = 10 if mode == "weekly" else 100
    print(
        f"\nDeepResearch Bench ({mode}, {n_questions} questions):\n"
        f"  Estimated cost: ${low}-{high} (RACE judge: GPT-5.5, FACT judge: GPT-5.4-mini,\n"
        f"  plus this project's own agent execution cost — see docs/DESIGN.md §5.6).\n"
        f"  docs/DESIGN.md decision row 11: this is NOT nightly-affordable, and the\n"
        f"  full 100-task suite is monthly/manual only, never automated.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["weekly", "full"], default="weekly")
    parser.add_argument("--confirm", action="store_true", help="required to proceed past the cost estimate")
    args = parser.parse_args()

    print_cost_estimate(args.mode)

    if not args.confirm:
        print("Not running — pass --confirm to proceed past this cost estimate.")
        sys.exit(1)

    raise NotImplementedError(
        "The RACE/FACT judge pipeline (Ayanami0730/deep_research_bench) isn't wired up yet — "
        "this session built FRAMES + MuSiQue in full and left DeepResearch Bench as this gated "
        "stub, per its lower priority in the task list. See docs/RESULTS.md."
    )


if __name__ == "__main__":
    main()
