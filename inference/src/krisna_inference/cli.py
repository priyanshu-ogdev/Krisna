"""Krisna Inference CLI: Harness for running agentic design sessions.

Provides the CLI entry point for the Krisna Swap Orchestrator, allowing
operators to run conversational turns, finalize renders, and critique designs
either interactively or non-interactively using MockBackend or real model backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

# Ensure monorepo packages are importable when running without editable install
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _pkg in (_REPO_ROOT / "inference" / "src", _REPO_ROOT / "training" / "src"):
    if _pkg.exists() and str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))



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
    """Execute an agentic session through the SwapOrchestrator."""
    if args.real:
        # Deferred import — MockBackend mode (default) works with zero real-backend dependencies
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
        print(
            "(MockBackend mode: this exercises the real flow shape and orchestration\n"
            " logic, but NOT real model output — pass --real with a GPU and models/\n"
            " checkpoints configured via KRISNA_* env vars to test actual generations.)"
        )

    await orch.start()
    print(f"Baseline resident: {orch.ledger.snapshot()}")

    try:
        messages = args.message or ["a minimalist login screen for a banking app"]
        turn_result = None
        conversation_history: list[dict] = []
        for i, message in enumerate(messages, 1):
            _print_header(f"Conversational turn {i}/{len(messages)}: {message!r}")
            turn_result = await orch.run_conversational_turn(
                message=message,
                constraints=args.constraints or {},
                conversation_history=conversation_history,
            )
            _print_result("conversational_turn", turn_result)
            conversation_history.append({"role": "user", "content": message})
            reply = (turn_result.get("planner") or {}).get("reply_text") if isinstance(turn_result, dict) else None
            if reply:
                conversation_history.append({"role": "planner", "content": str(reply)})

        if args.finalize:
            _print_header(f"Finalize (preferred_tier={args.finalize_tier})")
            finalize_prompt = messages[-1] if messages else None
            finalize_result = await orch.request_finalize(
                preferred_tier=args.finalize_tier,
                prompt=finalize_prompt,
                constraints=args.constraints or {},
            )
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
    """Build the CLI argument parser for agentic sessions."""
    p = argparse.ArgumentParser(
        prog="krisna-session",
        description="Krisna Inference Agentic Session CLI harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""
    args = build_parser().parse_args(argv)

    from krisna_inference.orchestrator.model_registry import Tier

    tier_map = {
        "polish_default": Tier.POLISH_DEFAULT,
        "polish_quality": Tier.POLISH_QUALITY,
    }
    args.finalize_tier = tier_map[args.finalize_tier]

    return asyncio.run(run_session(args))


if __name__ == "__main__":
    sys.exit(main())
