from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepresearch.agent.orchestrator import run_research  # noqa: E402
from deepresearch.backends import build_search_backend  # noqa: E402
from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.telemetry.otel_setup import init_telemetry  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "hand_picked_questions.json"
TRAJECTORIES_DIR = Path(__file__).parent.parent / "trajectories"


async def main() -> None:
    init_telemetry()
    questions = json.loads(QUESTIONS_FILE.read_text())
    TRAJECTORIES_DIR.mkdir(exist_ok=True)
    config = RunConfig()
    backend = build_search_backend(config)

    for item in questions:
        print(f"Running {item['id']}: {item['question']}")
        result = await run_research(item["question"], config=config, search_backend=backend)
        out_path = TRAJECTORIES_DIR / f"{item['id']}.json"
        out_path.write_text(json.dumps(result.model_dump(), indent=2))
        print(f"  status={result.status.value} cost=${result.total_cost_usd:.4f} -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
