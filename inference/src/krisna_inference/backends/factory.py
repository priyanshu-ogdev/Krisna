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
import sys

from krisna_inference.orchestrator.model_registry import ModelBackend, ModelSpec, Tier

# Low-VRAM / CPU-offload mode. When set, real_backend_factory passes
# offload params to the backends that support it (POLISH_DEFAULT,
# POLISH_QUALITY, CRITIC — PLANNER/SKETCH are small enough not to need
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
        from krisna_inference.backends.planner_backend import PlannerBackend

        # REMOVED: KRISNA_PLANNER_LORA_PATH. The Planner ships frozen —
        # RAG over data-forge's real UICrit critique corpus + constrained
        # JSON decoding, no fine-tune. See planner_backend.py.
        return PlannerBackend(
            spec,
            use_fast_mode=os.environ.get("KRISNA_PLANNER_FAST_MODE", "0") == "1",
            rag_corpus_dir=os.environ.get("KRISNA_PLANNER_RAG_CORPUS_DIR"),
            # Not wired to _LOW_VRAM — the Planner's 6.5GB NF4 footprint
            # already fits a 12GB target on its own (see
            # LOW_VRAM_REGISTRY), so offloading it would only add latency
            # for no VRAM benefit.
        )

    if spec.tier == Tier.SKETCH:
        from krisna_inference.backends.sketch_backend import SketchBackend

        return SketchBackend(
            spec,
            checkpoint_path=os.environ.get("KRISNA_SKETCH_CHECKPOINT"),
            guidance_scale=float(os.environ.get("KRISNA_SKETCH_GUIDANCE_SCALE", "3.0")),
        )

    if spec.tier == Tier.POLISH_DEFAULT:
        from krisna_inference.backends.polish_default_backend import ZImageTurboBackend

        # Z-Image-Turbo is the ONE renderer this project actually fine-tunes
        # (LoRA + Diffusion-DPO per the final PRD) — this LoRA wiring is
        # correct and must stay, unlike the three removed above/below.
        # WAS "Not wired to _LOW_VRAM — 8.0GB already fits a 12GB target."
        # That 8.0GB figure assumed NF4 quantization this backend never
        # applies (bf16 only, by design — see that file's load()). Real
        # bf16 footprint is ~14GB, which does NOT already fit a 12GB
        # target, so this now wires the same enable_cpu_offload mechanism
        # Polish Quality uses. See model_registry.py's LOW_VRAM_REGISTRY
        # entry for this tier and docs/review/13_ram_offload_and_precision_audit.md.
        return ZImageTurboBackend(
            spec,
            lora_adapter_path=os.environ.get("KRISNA_POLISH_DEFAULT_LORA_PATH"),
            enable_cpu_offload=_LOW_VRAM,
        )

    if spec.tier == Tier.POLISH_QUALITY:
        from krisna_inference.backends.polish_quality_backend import QwenImageEditBackend

        # REMOVED: KRISNA_POLISH_QUALITY_LORA_PATH. Qwen-Image-Edit-2511
        # ships frozen — zero-shot ICL edit conditioning + SDEdit-style
        # partial denoising at inference time. See polish_quality_backend.py.
        return QwenImageEditBackend(
            spec,
            edit_strength=float(os.environ.get("KRISNA_POLISH_QUALITY_EDIT_STRENGTH", "0.65")),
            # enable_model_cpu_offload() ONLY — never
            # enable_sequential_cpu_offload(), which has a confirmed
            # incompatibility with bnb NF4 (diffusers GH issue #10800).
            enable_cpu_offload=_LOW_VRAM,
        )

    if spec.tier == Tier.CRITIC:
        from krisna_inference.backends.critic_backend import CriticBackend

        # REMOVED: KRISNA_CRITIC_LORA_PATH. Gemma 4 ships frozen — an
        # on-demand product feature, never trained by this project. See
        # critic_backend.py / critic_worker.py.
        max_gpu_gb = None
        if _LOW_VRAM:
            max_gpu_gb = float(os.environ.get("KRISNA_CRITIC_MAX_GPU_GB", "11.5"))
        default_worker_python = (
            "./venv-critic/Scripts/python.exe" if sys.platform == "win32" else "./venv-critic/bin/python"
        )
        return CriticBackend(
            spec,
            worker_python=os.environ.get("KRISNA_CRITIC_VENV_PYTHON", default_worker_python),
            # When set, critic_worker.py bypasses unsloth's FastModel (no
            # verified CPU-offload support) for plain transformers+bnb
            # instead — see that module's _load() docstring, including the
            # real ~40GB system-RAM cost this implies for Gemma 4 31B.
            max_gpu_gb=max_gpu_gb,
            max_cpu_gb=float(os.environ.get("KRISNA_CRITIC_MAX_CPU_GB", "64")) if _LOW_VRAM else None,
        )

    raise ValueError(f"No real backend wired for tier: {spec.tier}")
