#!/usr/bin/env python3
"""Critic worker — runs Gemma 4 31B Dense in a SEPARATE Python environment
from the rest of krisna_inference (formerly krisna_orchestrator).

FROZEN MODEL (final, no-RLHF-loop PRD revision): Gemma 4 ships as an
on-demand, product-side critique feature only. It is never trained by
this project — no LoRA/QLoRA adapter is loaded here, and none should be.
There is no AI-judge-labeled training data generated anywhere in this
pipeline; this worker's output is a live product feature, not a data
source for any training job.

Why this is still a separate process — UPDATED reasoning (this review pass
closes out `docs/review/27_prd_open_risks_research.md`'s own "not yet
executed" item): the ORIGINAL justification here was that the Planner
needed a `transformers` built from git main for Qwen3.5 support, which
was true when this was written but has since been resolved — Qwen3.5 is
natively supported in `transformers>=5.2.0`, a tagged PyPI release (Feb
2026), confirmed via two independent sources (see
`docs/architecture/RESEARCH_AND_CITATIONS.md` and
`docs/review/27_prd_open_risks_research.md` §2.1). So the ORIGINAL
version-range conflict this split was built to route around no longer
exists — `transformers>=5.2.0` and `transformers==5.5.0` are not in
conflict, and diffusers' own actual floor (verified directly from its
setup.py: `transformers>=4.41.2`, no upper bound) doesn't conflict with
either.

The split is still correct and still necessary, but for a DIFFERENT,
more specific reason, found during this pass: `transformers==5.5.0` is
not an arbitrary compatible pin — it is the last version before a
confirmed regression in bnb-4bit dequantization for Unsloth's
prequantized checkpoints (exactly what this tier loads). Verified via
unslothai/unsloth-zoo PR #1227's own measurement: on real hardware,
`unsloth/gemma-4-31B-it-unsloth-bnb-4bit` loads with all 352/352 modules
correctly dequantized (`quant_state` present) on transformers 5.5.0, but
0/352 on transformers 5.17.0 — a real forward-pass-breaking defect,
tracked upstream as unslothai/unsloth#9867, #10010, #10017, #10276. So
holding this tier at exactly 5.5.0 is a deliberate regression-avoidance
pin, not a stale/lazy one — and it's specifically because of THIS pin
that the split earns its keep: it lets the other three tiers (Planner,
Sketch, Polish) move forward onto newer transformers/diffusers releases
over time without being held back by a Critic-specific regression that
has nothing to do with them, and without forcing Critic to eat whatever
transformers changes land for everyone else. Collapsing this back into
one venv would mean either every tier is capped at 5.5.0 forever, or
Critic silently breaks the next time someone bumps the shared pin for
an unrelated reason — this worker/subprocess split is what prevents
that from ever happening silently. Concretely: this worker runs in its
own venv (see requirements-critic.txt) and talks to the main
orchestrator process over stdin/stdout JSON lines — the same shape as the
vLLM-subprocess pattern already used in the data-forge project's engine.py.

Protocol: one JSON object per line on stdin, one JSON object per line on
stdout. Commands: {"cmd": "load"} / {"cmd": "run", ...} / {"cmd": "unload"}.
Every response includes {"ok": bool, ...}. An OOM is reported as
{"ok": false, "error_type": "oom", "error": "..."} so the parent process
can distinguish it from other load failures without parsing text.
"""

from __future__ import annotations

import json
import sys
import traceback

MODEL_ID = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"  # pre-quantized NF4, per §6

# CRITIQUE_SOURCE is the single place this worker's identity string lives —
# every emitted critique_result uses this constant rather than a repeated
# literal, so there's exactly one spot to update if the model/quant changes.
# Renamed from the old "gemma4_31b_qlora" (which implied a trained QLoRA
# adapter is applied — it never is, under the final PRD): this model is
# frozen, so the name should say so rather than imply the opposite.
CRITIQUE_SOURCE = "gemma4_31b_frozen"

_model = None
_tokenizer = None


def _load(params: dict) -> dict:
    global _model, _tokenizer
    max_gpu_gb = params.get("max_gpu_gb")

    if max_gpu_gb is not None:
        # Low-VRAM / CPU-offload mode. unsloth's FastModel.from_pretrained
        # has NO documented/verified max_memory or CPU-offload support
        # (checked before writing this — its own bug tracker shows users
        # hitting plain OOM on VRAM-constrained cards rather than a
        # graceful offload path). Rather than assume it works, this
        # bypasses unsloth entirely for this mode and uses plain
        # transformers+bitsandbytes instead — the same verified pattern
        # planner_backend.py uses. Real, correct, but loses unsloth's
        # inference speedup; that's a documented trade-off of low-VRAM
        # mode, not a bug.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id = params.get("model_id", MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # CONFIRMED real, documented requirement: bitsandbytes
                # raises "Some modules are dispatched on the CPU or the
                # disk..." unless this flag is set alongside max_memory.
                # Real cost: the CPU-offloaded portion is stored in FP32,
                # not 4-bit — an 8x per-parameter size increase relative
                # to NF4. For Gemma 4 31B (~30.7B params), offloading
                # enough to hit a ~12GB GPU target means roughly 40GB of
                # system RAM for the offloaded portion — see
                # model_registry.py's LOW_VRAM_REGISTRY comment for the
                # full calculation this number comes from.
                llm_int8_enable_fp32_cpu_offload=True,
            ),
            max_memory={0: f"{max_gpu_gb}GiB", "cpu": f"{params.get('max_cpu_gb', 64)}GiB"},
        )
        model.eval()
        _model, _tokenizer = model, tokenizer
        return {"ok": True, "offload_mode": True}

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=params.get("model_id", MODEL_ID),
        max_seq_length=params.get("max_seq_length", 8192),
        load_in_4bit=True,  # NF4, already baked into the -unsloth-bnb-4bit checkpoint
    )
    # REMOVED: LoRA adapter loading (PeftModel.from_pretrained). Gemma 4
    # ships frozen — see module docstring. If a future PRD revision
    # un-freezes this model, restore adapter loading here explicitly
    # rather than silently reintroducing it; do not assume the old
    # lora_adapter_path plumbing elsewhere in this repo (factory.py,
    # training/critic/) is safe to wire back in as-is without re-reading
    # this docstring's reasoning first.

    model.eval()
    _model, _tokenizer = model, tokenizer
    return {"ok": True, "offload_mode": False}


def _unload(params: dict) -> dict:
    global _model, _tokenizer
    import gc

    _model = None
    _tokenizer = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"ok": True}


def _build_prompt(constraints: dict) -> str:
    """Factored out so tests can catch drift against
    training/critic/dataset.py's `build_critic_prompt()` — that copy MUST
    stay word-for-word identical to this one (see that module's
    docstring for why it's a duplicate rather than an import: this script
    must stay standalone, runnable via subprocess with no
    krisna_inference (formerly krisna_orchestrator) import at all)."""
    return (
        "You are a senior UI/graphic design critic. Judge the ATTACHED "
        "finished design render against these constraints: "
        f"{json.dumps(constraints)}. Respond with ONLY a JSON object matching "
        "this exact shape: {\"overall_score\": <0-1 float>, \"dimensions\": "
        "{\"visual_hierarchy\": {\"score\": <0-1>, \"note\": \"<=2 sentences\"}, "
        "\"readability\": {...}, \"layout_consistency\": {...}, "
        "\"brand_alignment\": {...}}, \"suggested_edits\": [{\"region\": "
        "[x,y,w,h], \"instruction\": \"...\"}]}"
    )


def _run(params: dict) -> dict:
    import torch
    from PIL import Image

    if _model is None:
        return {"ok": False, "error_type": "not_loaded", "error": "Critic model not loaded"}

    image_path = params["image_path"]  # resolved path on shared/mounted disk, not a blob:// ref
    constraints = params.get("constraints", {})
    image = Image.open(image_path)

    prompt = _build_prompt(constraints)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(_model.device)

    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=1024, temperature=0.3, do_sample=True)
    raw_text = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    try:
        start, end = raw_text.index("{"), raw_text.rindex("}") + 1
        parsed = json.loads(raw_text[start:end])
    except (ValueError, json.JSONDecodeError):
        # Model didn't return clean JSON — surface the raw text rather than
        # crashing the worker; the Critique Adapter on the parent side can
        # decide how to handle a malformed critique.
        return {
            "ok": True,
            "critique_result": {
                "critique_source": CRITIQUE_SOURCE,
                "overall_score": 0.0,
                "dimensions": {},
                "suggested_edits": [],
                "raw_model_output_ref": raw_text[:2000],
            },
        }

    critique_result = {
        "critique_source": CRITIQUE_SOURCE,
        "overall_score": parsed.get("overall_score", 0.0),
        "dimensions": parsed.get("dimensions", {}),
        "suggested_edits": parsed.get("suggested_edits", []),
        "raw_model_output_ref": None,
    }
    return {"ok": True, "critique_result": critique_result}


HANDLERS = {"load": _load, "run": _run, "unload": _unload}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handler = HANDLERS[request["cmd"]]
            response = handler(request.get("params", {}))
        except Exception as e:  # noqa: BLE001 - worker must never crash silently
            msg = str(e)
            error_type = "oom" if "out of memory" in msg.lower() else "exception"
            response = {
                "ok": False,
                "error_type": error_type,
                "error": msg,
                "traceback": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
