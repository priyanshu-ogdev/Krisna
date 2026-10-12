#!/usr/bin/env python3
"""Evaluate Planner contract/runtime behavior on JSONL cases.

Each input line must contain message and optionally constraints, history, and
expected fields. The runner intentionally records raw outputs and timings so
quantization/runtime bake-offs remain comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


async def main_async(args):
    from krisna_inference.backends.factory import real_backend_factory
    from krisna_inference.orchestrator.model_registry import REGISTRY, Tier

    backend = real_backend_factory(REGISTRY[Tier.PLANNER])
    await backend.load()
    rows = []
    try:
        for line in Path(args.cases).read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            started = time.perf_counter()
            error = None
            output = None
            try:
                output = await backend.run(**case)
            except Exception as exc:  # preserve failures in the report
                error = f"{type(exc).__name__}: {exc}"
            rows.append({"case_id": case.get("case_id"), "latency_ms": (time.perf_counter() - started) * 1000,
                         "error": error, "output": output})
    finally:
        await backend.unload()
    Path(args.output).write_text(json.dumps({"runtime": args.runtime, "results": rows}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cases")
    p.add_argument("--output", default="models/planner/eval.json")
    p.add_argument("--runtime", default="configured")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
