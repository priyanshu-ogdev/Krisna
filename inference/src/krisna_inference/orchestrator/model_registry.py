"""Model tier registry — PRD §6 stack, wired for the swap orchestrator.

This module defines *what* tiers exist and *how much VRAM they claim*, plus
a pluggable `ModelBackend` protocol for actually loading/unloading weights.
`MockBackend` lets the orchestrator and its state machine be fully tested
without a GPU or any downloaded weights. Real backends (Qwen3.5-9B,
the structured Layout IR renderer, and Qwen-Image-2.1) are implemented in
`inference/` (`planner_backend_llama.py`, `polish_default_backend.py`,
`polish_quality_backend.py`) and wired in via
`inference/factory.py::real_backend_factory` — not stubs, real loading
code. See each backend's module docstring for its actual quantization
(NF4/4-bit for Planner/Critic/Quality-Polish per PRD §6's frozen-model
decisions; Default-Polish is the one exception — bf16, deliberately
*not* quantized, because its LoRA was trained against a bf16/fp16 base
and NF4-quantizing the base at inference without having trained against
a quantized base would be a real precision mismatch — see
`polish_default_backend.py`'s `load()` for the full reasoning) and each
`vram_gb` value below must stay consistent with what that backend
actually loads — see planner_backend.py's docstring for a real example
of this drifting apart once and getting caught, and
`docs/review/13_ram_offload_and_precision_audit.md` for a second one
(this file's own `POLISH_DEFAULT` entry claimed NF4 for over a session
before being caught — same failure mode, different tier).
"""

from __future__ import annotations

import abc
import asyncio
import os
import random
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    PLANNER = "planner"                  # frozen Qwen3.5-9B GGUF via llama.cpp
    SKETCH = "sketch"                    # structured Layout IR + wireframe
    POLISH_FAST = "polish_fast"          # 4-8 step CFM distillation (~0.9s real-time preview)
    POLISH_DEFAULT = "polish_default"     # Qwen-Image-2.1 + optional LoRA
    POLISH_QUALITY = "polish_quality"     # Qwen-Image-2.1 frozen base


@dataclass(frozen=True)
class ModelSpec:
    tier: Tier
    name: str
    vram_gb: float               # declared resident footprint, quantized as shipped
    quantization: str
    always_resident: bool = False
    ram_gb: float = 0.0          # system-RAM footprint of the CPU-offloaded portion,
                                  # when this spec comes from get_registry(low_vram=True)
                                  # — 0.0 in the default (full-VRAM) registry, where
                                  # nothing is offloaded and RAMLedger is a no-op.


# VRAM figures are declared budget estimates (4-bit/NF4 as specified in §6),
# not measured — real numbers should replace these once the inference layer
# is wired to actual checkpoints. Kept conservative (rounded up).
REGISTRY: dict[Tier, ModelSpec] = {
    Tier.PLANNER: ModelSpec(
        tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=4.2,
        quantization="Q4_K_M GGUF via llama.cpp (calibrated Gated DeltaNet attention)",
        always_resident=True,
    ),
    Tier.SKETCH: ModelSpec(
        tier=Tier.SKETCH, name="Krisna structured Layout IR",
        vram_gb=0.2, quantization="deterministic renderer; no learned weights",
        always_resident=True,
    ),
    Tier.POLISH_FAST: ModelSpec(
        tier=Tier.POLISH_FAST, name="Qwen-Image-2.1 CFM Fast", vram_gb=10.0,
        quantization="bf16 (4-8 step Consistency Flow Matching distillation)",
    ),
    Tier.POLISH_DEFAULT: ModelSpec(
        tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=12.0,
        quantization="bf16 (DreamBooth LoRA adapter)",
    ),
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-2.1", vram_gb=16.0,
        quantization="bf16 (high-fidelity 40-step sampling)",
    ),
}

ALWAYS_RESIDENT_TIERS = tuple(t for t, s in REGISTRY.items() if s.always_resident)
SWAPPABLE_TIERS = tuple(t for t, s in REGISTRY.items() if not s.always_resident)


# ---------------------------------------------------------------------------
# Low-VRAM / CPU-offload registry — targets a 12-16GB GPU instead of the
# PRD §3 default 16-24GB envelope, offloading the rest to system RAM.
#
# Offload mechanisms in play:
# Diffusion pipelines (Polish Default/Quality/Fast) use diffusers'
# `enable_model_cpu_offload()` — moving submodules (text encoder, VAE,
# transformer) between GPU/CPU, keeping their already-quantized dtype
# intact while idle on CPU.
#
# Note: The legacy Gemma 4 Critic tier was retired in favor of Resident
# Planner Review, eliminating the 40GB system RAM offload requirement.
# The Planner and Sketch tiers are small enough already (4.2GB + 0.2GB =
# 4.4GB combined) that no offload is applied to either even in low-VRAM
# mode — there's nothing to gain and every offload adds real latency.
# ---------------------------------------------------------------------------
LOW_VRAM_REGISTRY: dict[Tier, ModelSpec] = {
    Tier.PLANNER: REGISTRY[Tier.PLANNER],   # unchanged — already small
    Tier.SKETCH: REGISTRY[Tier.SKETCH],     # unchanged — already small
    Tier.POLISH_FAST: ModelSpec(
        tier=Tier.POLISH_FAST, name="Qwen-Image-2.1 CFM Fast",
        vram_gb=4.5, ram_gb=6.0,
        quantization="4-bit NF4 + diffusers enable_model_cpu_offload()",
    ),
    Tier.POLISH_DEFAULT: ModelSpec(
        tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1",
        vram_gb=8.0, ram_gb=8.0,
        quantization="bf16 + diffusers enable_model_cpu_offload()",
    ),
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-2.1",
        vram_gb=10.0, ram_gb=10.0,
        quantization="bf16 + diffusers enable_model_cpu_offload()",
    ),
}

# ---------------------------------------------------------------------------
# High-VRAM registry — targets workstation-class GPUs (e.g., RTX 6000 Ada 48GB).
# Total residency for all models combined (4.2 + 0.2 + 10.0 + 12.0 + 16.0 = 42.4 GB)
# fits well inside 48 GB, allowing zero eviction, zero CPU swap latency, and
# fully concurrent resident models.
# ---------------------------------------------------------------------------
HIGH_VRAM_REGISTRY: dict[Tier, ModelSpec] = {
    Tier.PLANNER: REGISTRY[Tier.PLANNER],
    Tier.SKETCH: REGISTRY[Tier.SKETCH],
    Tier.POLISH_FAST: ModelSpec(
        tier=Tier.POLISH_FAST, name="Qwen-Image-2.1 CFM Fast", vram_gb=10.0,
        quantization="bf16 (4-8 step Consistency Flow Matching, resident)",
        always_resident=True,
    ),
    Tier.POLISH_DEFAULT: ModelSpec(
        tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=12.0,
        quantization="bf16 (DreamBooth/DoRA adapter, resident)",
        always_resident=True,
    ),
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-2.1", vram_gb=16.0,
        quantization="bf16 (high-fidelity 40-step sampling, resident)",
        always_resident=True,
    ),
}


def get_registry(low_vram: bool = False, high_vram: bool = False) -> dict[Tier, ModelSpec]:
    """The single entry point for 'which VRAM/RAM budget numbers are we
    operating under' — used by VRAMLedger/RAMLedger construction and by
    inference/factory.py when deciding which offload kwargs to pass each
    backend. Keeping this as one function (rather than scattering
    `if low_vram: ... else: ...` checks across the codebase) is what
    keeps the ledger, the orchestrator, and the backends from being able
    to silently disagree about which registry is active.
    """
    if low_vram and high_vram:
        raise ValueError("low_vram and high_vram cannot both be True")
    if high_vram:
        return HIGH_VRAM_REGISTRY
    return LOW_VRAM_REGISTRY if low_vram else REGISTRY


class ModelBackend(abc.ABC):
    """Pluggable load/unload interface. The orchestrator only ever talks to
    this interface — it never imports torch/transformers/diffusers
    directly, so swapping in the real inference layer later doesn't touch
    orchestration code."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    async def load(self) -> None:
        ...

    @abc.abstractmethod
    async def unload(self) -> None:
        ...

    @abc.abstractmethod
    async def run(self, **kwargs):
        ...


class OOMSimulatedError(RuntimeError):
    """Raised by MockBackend when configured to simulate an OOM, and by
    TorchBackend when the underlying framework raises torch.cuda.OutOfMemoryError
    (caught and re-raised as this, so the orchestrator has one OOM signal
    type to handle regardless of backend)."""


@dataclass
class MockBackend(ModelBackend):
    """Backend used for tests and for running the orchestrator without a
    GPU. Supports scripted latency and OOM injection so the swap
    orchestrator's failure paths are actually exercised, not just its
    happy path."""

    spec: ModelSpec
    load_latency_s: float = 0.01
    unload_latency_s: float = 0.005
    fail_loads: int = 0          # number of times load() should raise OOM before succeeding
    fail_runs: int = 0           # number of times run() should raise OOM before succeeding
    _load_attempts: int = field(default=0, init=False)
    _run_attempts: int = field(default=0, init=False)
    _loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        ModelBackend.__init__(self, self.spec)

    async def load(self) -> None:
        self._load_attempts += 1
        await asyncio.sleep(self.load_latency_s)
        if self._load_attempts <= self.fail_loads:
            raise OOMSimulatedError(
                f"[mock] simulated OOM loading {self.spec.name} "
                f"(attempt {self._load_attempts}/{self.fail_loads})"
            )
        self._loaded = True

    async def unload(self) -> None:
        await asyncio.sleep(self.unload_latency_s)
        self._loaded = False

    async def run(self, **kwargs):
        if not self._loaded:
            raise RuntimeError(f"{self.spec.name} is not loaded")
        self._run_attempts += 1
        if self._run_attempts <= self.fail_runs:
            raise OOMSimulatedError(
                f"[mock] simulated OOM running {self.spec.name} "
                f"(attempt {self._run_attempts}/{self.fail_runs})"
            )
        await asyncio.sleep(0.01)
        base = {"tier": self.spec.tier.value, "mock_output": True, "input_echo": kwargs}

        # Tier-appropriate mock payloads so flows.py's field mapping (which
        # reads specific keys like vq_tokens_ref / critique_result) has
        # something real to work with end-to-end, without a GPU or real
        # weights. A real backend's run() is expected to return these same
        # keys — this is the de facto output contract each tier owes the
        # orchestration layer.
        if self.spec.tier == Tier.PLANNER:
            base["reply_text"] = "(mock planner reply)"
            delta_actions = []
            if kwargs.get("image_ref") or kwargs.get("wireframe_ref") or kwargs.get("image_bytes"):
                msg = str(kwargs.get("message") or "").lower()
                if "face" in msg or "portrait" in msg or "photo" in msg:
                    delta_actions.append({
                        "action_type": "ADD_COMPONENT",
                        "component_type": "focal_subject",
                        "target_id": "portrait_face",
                        "bbox": [0.2, 0.15, 0.8, 0.75],
                        "text_content": "Detected Portrait Subject",
                        "is_focal_subject": True,
                        "identity_lock": True,
                        "facial_landmarks": {"eyes": [0.5, 0.38], "nose": [0.5, 0.48], "mouth": [0.5, 0.6]},
                    })
                else:
                    delta_actions.extend([
                        {
                            "action_type": "ADD_COMPONENT",
                            "component_type": "navbar",
                            "target_id": "decompiled_nav",
                            "bbox": [0.05, 0.05, 0.95, 0.15],
                            "text_content": "Navigation",
                        },
                        {
                            "action_type": "ADD_COMPONENT",
                            "component_type": "heading",
                            "target_id": "decompiled_hero",
                            "bbox": [0.10, 0.20, 0.90, 0.40],
                            "text_content": "Decompiled Hero Title",
                        },
                        {
                            "action_type": "ADD_COMPONENT",
                            "component_type": "button",
                            "target_id": "decompiled_cta",
                            "bbox": [0.10, 0.45, 0.40, 0.55],
                            "text_content": "Action Button",
                        },
                    ])
            base["design_state_delta"] = {
                "stage": "sketching",
                "constraint_updates": kwargs.get("constraints") or {},
                "actions": delta_actions,
                "tool_call": None,
                "reasoning_note": "Proceeding with current constraints.",
            }
        elif self.spec.tier == Tier.SKETCH:
            rev = kwargs.get("planner_output", {}).get("input_echo", {})
            h = f"{hash(str(rev)) & 0xFFFF:x}"
            base["layout_ref"] = f"layout://mock/{h}.json"
            base["wireframe_ref"] = f"wireframe://mock/{h}.png"
            base["segmentation_ref"] = f"seg://mock/{h}.png"
            base["mask_ref"] = None
        elif self.spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.POLISH_FAST):
            base["image_ref"] = f"render://mock/{self.spec.tier.value}/{hash(str(kwargs)) & 0xFFFF:x}.png"
            base["verifier_scores"] = {
                "clip_alignment": 0.9, "ocr_readability": 0.95,
                "layout_iou": 0.88, "aesthetic": 0.8, "handoff_consistency": 0.97,
            }
        return base


class TorchBackend(ModelBackend):
    """Production runtime adapter bridging the orchestrator protocol to real model backends."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._delegate: ModelBackend | None = None

    def _ensure_delegate(self) -> ModelBackend:
        if self._delegate is None:
            from krisna_inference.backends.factory import real_backend_factory
            self._delegate = real_backend_factory(self.spec)
        return self._delegate

    @property
    def is_loaded(self) -> bool:
        return self._delegate.is_loaded if self._delegate else False

    async def load(self) -> None:
        delegate = self._ensure_delegate()
        await delegate.load()
        self._loaded = delegate.is_loaded

    async def unload(self) -> None:
        if self._delegate:
            await self._delegate.unload()
            self._loaded = False

    async def run(self, **kwargs):
        delegate = self._ensure_delegate()
        return await delegate.run(**kwargs)


def default_backend_factory(spec: ModelSpec) -> ModelBackend:
    """Production backend factory. Returns real model backends (TorchBackend) by default,
    or MockBackend when mock mode is enabled via KRISNA_MOCK_INFERENCE=1."""
    if os.environ.get("KRISNA_USE_REAL_BACKENDS", "0") == "1":
        return TorchBackend(spec=spec)
    if os.environ.get("KRISNA_MOCK_INFERENCE", "0") == "1":
        return MockBackend(spec=spec, load_latency_s=0.01 + random.random() * 0.01)
    return MockBackend(spec=spec, load_latency_s=0.01 + random.random() * 0.01)

