#!/usr/bin/env python3
"""Inference-runtime: a CLI harness for running a real agentic session
end-to-end against checkpoints in models/ — for manually testing trained
models, not a production entry point (that's
scripts/inference/run_service.sh + the FastAPI service).

What this actually does: constructs a real SwapOrchestrator (MockBackend
by default — safe, no GPU needed; --real for actual model backends), runs
it through the PRD §5.3 sequence flows in order — conversational turn(s),
finalize, optional critique — and prints the real result of each step, so
you can eyeball whether a freshly-trained checkpoint behaves sanely
without standing up the full FastAPI service and a client to talk to it.

This is deliberately a thin CLI over the SAME orchestrator class the real
service uses (krisna_inference.orchestrator.swap_orchestrator.
SwapOrchestrator) — not a separate, parallel implementation of the
agentic flow that could silently drift from what actually ships. If this
script's behavior differs from the service's, that's a bug in one of the
two, not an intentional difference.

Usage:
    # Safe default — MockBackend, no GPU, no models/ checkpoints needed.
    # Exercises the full flow shape without claiming to test real output.
    python inference-runtime/run_agentic_session.py

    # Real backends — GPU required, reads models/ checkpoints via the
    # same KRISNA_* env vars the real service uses.
    python inference-runtime/run_agentic_session.py --real

    # Real backends, low-VRAM/CPU-offload mode (see docs/inference/).
    python inference-runtime/run_agentic_session.py --real --low-vram

    # Scripted, non-interactive session (for CI/smoke-testing a checkpoint).
    python inference-runtime/run_agentic_session.py \\
        --message "a minimalist login screen" --finalize --critique
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _print_header(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def _print_result(label: str, result) -> None:
    print(f"\n[{label}]")
    if hasattr(result, "ok"):
        print(f"  ok:         {result.ok}")
        print(f"  tier_used:  {result.tier_used}")
        print(f"  degraded:   {result.degraded}")
        if result.error:
            print(f"  error:      {result.error}")
        print(f"  attempts:   {result.attempts}")
        if result.output is not None:
            print(f"  output:     {_short(result.output)}")
    else:
        print(f"  {_short(result)}")


def _short(obj, limit: int = 400) -> str:
    try:
        s = json.dumps(obj, default=str, indent=2)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "... [truncated]"


async def run_session(args: argparse.Namespace) -> int:
    if args.real:
        # Deferred import — MockBackend mode (the default) must work with
        # zero real-backend dependencies installed at all.
        from krisna_inference.backends.factory import real_backend_factory as backend_factory
    else:
        from krisna_inference.orchestrator.model_registry import default_backend_factory as backend_factory

    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator

    envelope_gb = args.vram_envelope_gb or (12.0 if args.low_vram else 24.0)
    orch = SwapOrchestrator(
        backend_factory=backend_factory,
        low_vram=args.low_vram,
        envelope_gb=envelope_gb,
        ram_envelope_gb=args.ram_envelope_gb,
    )

    _print_header(
        f"Starting orchestrator — backend={'REAL' if args.real else 'MockBackend'}, "
        f"registry={'low_vram' if args.low_vram else 'full_vram'}, "
        f"envelope={envelope_gb}GB"
    )
    if not args.real:
        print("(MockBackend mode: this exercises the real flow shape and orchestration\n"
              " logic, but NOT real model output — pass --real with a GPU and models/\n"
              " checkpoints configured via KRISNA_* env vars to test actual generations.)")

    await orch.start()
    print(f"Baseline resident: {orch.ledger.snapshot()}")

    try:
        messages = args.message or ["a minimalist login screen for a banking app"]
        turn_result = None
        for i, message in enumerate(messages, 1):
            _print_header(f"Conversational turn {i}/{len(messages)}: {message!r}")
            turn_result = await orch.run_conversational_turn(message=message, constraints=args.constraints or {})
            _print_result("conversational_turn", turn_result)

        if args.finalize:
            _print_header(f"Finalize (preferred_tier={args.finalize_tier})")
            finalize_result = await orch.request_finalize(preferred_tier=args.finalize_tier)
            _print_result("finalize", finalize_result)
            if not finalize_result.ok:
                print("\nFinalize failed — see error above. Not attempting critique.", file=sys.stderr)
                return 1

        if args.critique:
            _print_header("Critique")
            critique_result = await orch.request_critique()
            _print_result("critique", critique_result)
            if not critique_result.ok:
                print("\nCritique failed — see error above.", file=sys.stderr)
                return 1

        _print_header("Session complete")
        print(f"Final resident state: {orch.ledger.snapshot()}")
        print(f"RAM ledger: {orch.ram_ledger.snapshot()}")
        return 0

    finally:
        await orch.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--real", action="store_true",
        help="Use real model backends (GPU + models/ checkpoints required). "
             "Default: MockBackend — safe, tests orchestration flow shape only.",
    )
    p.add_argument(
        "--low-vram", action="store_true",
        help="Use the low-VRAM/CPU-offload registry (see docs/inference/). "
             "Only meaningful with --real.",
    )
    p.add_argument("--vram-envelope-gb", type=float, default=None, help="Override the VRAM envelope.")
    p.add_argument("--ram-envelope-gb", type=float, default=48.0, help="System-RAM budget for low-VRAM mode.")
    p.add_argument(
        "--message", action="append",
        help="A conversational turn message. Repeatable for a multi-turn session. "
             "Default: one turn with a placeholder message.",
    )
    p.add_argument("--constraints", type=json.loads, default=None, help="JSON dict of design constraints.")
    p.add_argument("--finalize", action="store_true", help="Run a finalize step after the conversational turn(s).")
    p.add_argument(
        "--finalize-tier", choices=["polish_default", "polish_quality"], default="polish_default",
        help="Which polish tier to request for finalize.",
    )
    p.add_argument("--critique", action="store_true", help="Run a critique step after finalize.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Resolve the --finalize-tier string into a real Tier enum member only
    # after argparse has validated it's one of the two real choices —
    # deferred import so --help works without krisna_inference installed
    # at all in a truly minimal environment.
    from krisna_inference.orchestrator.model_registry import Tier

    args.finalize_tier = {
        "polish_default": Tier.POLISH_DEFAULT,
        "polish_quality": Tier.POLISH_QUALITY,
    }[args.finalize_tier]

    return asyncio.run(run_session(args))


if __name__ == "__main__":
    sys.exit(main())
