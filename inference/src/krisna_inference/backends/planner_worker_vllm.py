#!/usr/bin/env python3
"""Planner worker (vLLM backend) — runs Qwen3.5 under vLLM in a SEPARATE
Python environment from the rest of krisna_inference.

See planner_backend_vllm.py's module docstring and
docs/review/29_vllm_planner_migration_research.md for the full research
trail on why this needs its own venv (vllm==0.29.0 pins torch==2.13.0
exactly — verified directly from its wheel metadata — a much stricter
constraint than the main venv's torch>=2.6.0) and why this whole path is
opt-in (KRISNA_PLANNER_BACKEND=vllm), not the default: it is verified at
the package/metadata level only, never against real GPU hardware in this
project's own environment.

Protocol: one JSON object per line on stdin, one JSON object per line on
stdout. Commands: {"cmd": "load", "params": {...}} / {"cmd": "generate",
"params": {"messages": [...], "max_tokens": int, "temperature": float}} /
{"cmd": "unload"}. Every response includes {"ok": bool, ...}. Mirrors
critic_worker.py's protocol shape exactly (same shape as the vLLM-
subprocess pattern already used in the data-forge project's engine.py).

Deliberately NO krisna_inference import anywhere in this file — same
reasoning as critic_worker.py: this script must stay standalone,
runnable via subprocess with no shared package dependency on the parent
process's environment. This is why the "generate" command takes
already-built chat `messages` rather than raw (constraints,
conversation_history, message) — RAG retrieval and system-prompt
construction (planner_backend.py's _build_system_prompt) stay in the
PARENT process, which already has krisna_inference importable and where
planner_rag.py's TF-IDF retriever lives (verified stdlib-only, no torch/
vllm — safe to keep there rather than duplicating it here too). This
worker's only job is: given fully-formed messages, generate text.
JSON-delta extraction and the retry loop ALSO stay in the parent
(PlannerBackend.run(), inherited unchanged by PlannerBackendVLLM) for the
same reason — unlike critic_worker.py's _run, which does need its own
JSON extraction because Critic's on-demand single-shot critique call has
no equivalent "already have this logic in a shared parent-side class"
structure to reuse.
"""

from __future__ import annotations

import json
import sys
import traceback

_llm = None


def _load(params: dict) -> dict:
    global _llm
    model_id = params["model_id"]
    gpu_memory_utilization = params["gpu_memory_utilization"]

    from vllm import LLM

    _llm = LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=False,
    )
    return {"ok": True}


def _generate(params: dict) -> dict:
    if _llm is None:
        return {"ok": False, "error_type": "not_loaded", "error": "Planner (vLLM) not loaded"}

    from vllm import SamplingParams

    messages = params["messages"]
    sampling_params = SamplingParams(
        max_tokens=params.get("max_tokens", 512),
        temperature=params.get("temperature", 0.7),
    )
    outputs = _llm.chat([messages], sampling_params, use_tqdm=False)
    return {"ok": True, "text": outputs[0].outputs[0].text}


def _unload(params: dict) -> dict:
    global _llm
    _llm = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"ok": True}


HANDLERS = {"load": _load, "generate": _generate, "unload": _unload}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handler = HANDLERS[request["cmd"]]
            response = handler(request.get("params", {}))
        except Exception as e:
            msg = str(e).lower()
            error_type = "oom" if ("out of memory" in msg or "cuda oom" in msg) else "exception"
            response = {
                "ok": False,
                "error_type": error_type,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
