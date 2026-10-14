"""Real-backend factory — the swap-in replacement for
model_registry.default_backend_factory (which returns MockBackend).

Usage:
    from krisna_inference.backends.factory import real_backend_factory
    orchestrator = SwapOrchestrator(backend_factory=real_backend_factory)

Kept in its own module (not model_registry.py) so importing model_registry
— which core orchestrator code, tests, and the service all do — never
pulls in torch/diffusers/transformers/unsloth. Those imports only happen
inside each backend's load()/run(), and this factory module itself only
imports the backend *classes*, which don't import their heavy deps at
class-definition time either.
"""

from __future__ import annotations

import os

from krisna_inference.orchestrator.model_registry import ModelBackend, ModelSpec, Tier

# Low-VRAM / CPU-offload mode. When set, real_backend_factory passes
# offload params to the Polish backend. Planner/Layout are the consumer-GPU
# baseline and do not use CPU offload.
# it, see model_registry.py's LOW_VRAM_REGISTRY comment). This flag alone does
# NOT change which registry SwapOrchestrator uses — that's the separate
# `low_vram=True` constructor arg (see swap_orchestrator.py). Passing this
# env var without also constructing the orchestrator with `low_vram=True`
# would offload the backends correctly but still enforce the full 24GB
# envelope/budget bookkeeping — set both together in practice, see
# scripts/run_service.sh's KRISNA_LOW_VRAM_MODE handling.
_LOW_VRAM = os.environ.get("KRISNA_LOW_VRAM_MODE", "0") == "1"


def real_backend_factory(spec: ModelSpec) -> ModelBackend:
    if spec.tier == Tier.PLANNER:
        from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
        return PlannerBackendLlama(spec, rag_corpus_dir=os.environ.get("KRISNA_PLANNER_RAG_CORPUS_DIR"))

    if spec.tier == Tier.SKETCH:
        from krisna_inference.backends.layout_backend import LayoutBackend
        return LayoutBackend(spec)

    if spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY):
        from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend

        model_id = os.environ.get("KRISNA_POLISH_MODEL_ID", "Qwen/Qwen-Image-2.1")
        lora_path = os.environ.get("KRISNA_POLISH_DEFAULT_LORA_PATH") if spec.tier == Tier.POLISH_DEFAULT else None
        # Qwen-Image-2.1's native pipeline is not SDEdit: do not configure
        # legacy strength/guidance or custom Euler/RK2 controls on a real
        # backend.  Synthetic compatibility callers may still pass those
        # controls directly to QwenImage21Backend with strict=False.
        strength = 1.0
        steps = 40 if spec.tier == Tier.POLISH_QUALITY else 28

        return QwenImage21Backend(
            spec,
            model_id=model_id,
            lora_adapter_path=lora_path,
            edit_strength=strength,
            num_inference_steps=steps,
            guidance_scale=1.0,
            image_guidance_scale=1.0,
            guidance_schedule="constant",
            solver_type="native",
            enable_cpu_offload=_LOW_VRAM,
            device="cuda",
            strict=True,
        )

    raise ValueError(f"No real backend wired for tier: {spec.tier}")
