from __future__ import annotations

import asyncio
import json
import sys

from deepresearch.agent.orchestrator import run_research
from deepresearch.backends import build_search_backend
from deepresearch.config import RunConfig
from deepresearch.telemetry.otel_setup import init_telemetry


async def _main(question: str) -> None:
    init_telemetry()
    config = RunConfig()
    backend = build_search_backend(config)
    result = await run_research(question, config=config, search_backend=backend)
    print(json.dumps(result.model_dump(), indent=2))
    if result.report:
        print("\n--- REPORT ---\n")
        print(result.report.text)


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m deepresearch.cli "<research question>"', file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(_main(sys.argv[1]))


if __name__ == "__main__":
    main()
