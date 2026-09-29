"""Model tier registry — PRD §6 stack, wired for the swap orchestrator.

This module defines *what* tiers exist and *how much VRAM they claim*, plus
a pluggable `ModelBackend` protocol for actually loading/unloading weights.
`MockBackend` lets the orchestrator and its state machine be fully tested
without a GPU or any downloaded weights. Real backends (Qwen3.5-9B,
Z-Image-Turbo, Qwen-Image-Edit-2511, Gemma 4) are implemented in
`inference/` (`planner_backend.py`, `polish_default_backend.py`,
`polish_quality_backend.py`, `critic_backend.py`) and wired in via
`inference/factory.py::real_backend_factory` — not stubs, real loading
code. See each backend's module docstring for its actual quantization
(NF4/4-bit for Planner/Critic/Quality-Polish per PRD §6's frozen-model
decisions, NF4/DPO-fine-tuned for Default-Polish) and each `vram_gb`
value below must stay consistent with what that backend actually loads —
see planner_backend.py's docstring for a real example of this drifting
apart once and getting caught.
"""

from __future__ import annotations

import abc
import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    PLANNER = "planner"                  # Qwen3.5-9B (fast mode: Qwen3.5-4B)
    SKETCH = "sketch"                     # UI-domain MaskGIT/MaskGIL lineage
    POLISH_DEFAULT = "polish_default"     # Z-Image-Turbo
    POLISH_QUALITY = "polish_quality"     # Qwen-Image-Edit-2511
    CRITIC = "critic"                     # Gemma 4 31B Dense


@dataclass(frozen=True)
class ModelSpec:
    tier: Tier
    name: str
    vram_gb: float               # declared resident footprint, quantized as shipped
    quantization: str
    always_resident: bool = False
    fallback_tier: Tier | None = None    # e.g. POLISH_QUALITY -> POLISH_DEFAULT on OOM
    ram_gb: float = 0.0          # system-RAM footprint of the CPU-offloaded portion,
                                  # when this spec comes from get_registry(low_vram=True)
                                  # — 0.0 in the default (full-VRAM) registry, where
                                  # nothing is offloaded and RAMLedger is a no-op.


# VRAM figures are declared budget estimates (4-bit/NF4 as specified in §6),
# not measured — real numbers should replace these once the inference layer
# is wired to actual checkpoints. Kept conservative (rounded up).
REGISTRY: dict[Tier, ModelSpec] = {
    Tier.PLANNER: ModelSpec(
        tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5,
        quantization="4-bit (fast mode: Qwen3.5-4B available)",
        always_resident=True,
    ),
    Tier.SKETCH: ModelSpec(
        tier=Tier.SKETCH, name="UI-domain MaskGIT/MaskGIL sketch tier",
        vram_gb=3.0, quantization="n/a (small, from-scratch)",
        always_resident=True,
    ),
    Tier.POLISH_DEFAULT: ModelSpec(
        tier=Tier.POLISH_DEFAULT, name="Z-Image-Turbo", vram_gb=8.0,
        quantization="NF4/NVFP4",
    ),
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-Edit-2511", vram_gb=16.0,
        # BUG FIX: was "NF4 (frozen backbone)" — that wording implies a
        # trainable adapter sits on top of a frozen quantized backbone
        # (the standard LoRA framing), when under the final PRD this
        # model has NO adapter at all — the whole pipeline runs frozen,
        # zero-shot ICL + SDEdit-style partial denoising only.
        quantization="NF4 (fully frozen — no adapter)",
        fallback_tier=Tier.POLISH_DEFAULT,
    ),
    Tier.CRITIC: ModelSpec(
        tier=Tier.CRITIC, name="Gemma 4 31B Dense", vram_gb=18.0,
        quantization="NF4 (fully frozen — on-demand product feature, never trained)",
    ),
}

ALWAYS_RESIDENT_TIERS = tuple(t for t, s in REGISTRY.items() if s.always_resident)
SWAPPABLE_TIERS = tuple(t for t, s in REGISTRY.items() if not s.always_resident)


# ---------------------------------------------------------------------------
# Low-VRAM / CPU-offload registry — targets a 12-16GB GPU instead of the
# PRD §3 default 16-24GB envelope, offloading the rest to system RAM.
#
# Two DIFFERENT, real offload mechanisms are in play here, with very
# different RAM-cost implications — conflating them would give a
# misleading picture of what this actually costs:
#
# 1. Diffusion pipelines (Polish Default/Quality) use diffusers'
#    `enable_model_cpu_offload()` — CONFIRMED safe with bnb NF4 (unlike
#    `enable_sequential_cpu_offload()`, which has a documented
#    incompatibility with bnb 4-bit: "Blockwise quantization only
#    supports 16/32-bit floats, but got torch.uint8" — diffusers GH
#    issue #10800). This moves whole submodules (text encoder, VAE,
#    transformer) between GPU/CPU, keeping their already-quantized dtype
#    intact while idle on CPU — vram_gb + ram_gb below sums to
#    approximately the original full-VRAM number, not more.
#
# 2. The Critic (Gemma 4, loaded via plain transformers+bitsandbytes when
#    offloading — NOT unsloth's FastModel, which has no documented/
#    verified CPU-offload support) uses `max_memory={0: "...GiB", "cpu":
#    "...GiB"}` + `llm_int8_enable_fp32_cpu_offload=True`. bitsandbytes'
#    own documented behavior for this path stores the CPU-resident
#    portion in FP32, not 4-bit — an 8x per-parameter size increase
#    relative to NF4. For Gemma 4 31B Dense (~30.7B params), offloading
#    enough to bring GPU residency down to ~12GB (roughly a third of the
#    ~18GB NF4 total) means ~10B params living on the CPU side at FP32:
#    10B params x 4 bytes = ~40GB of system RAM — NOT a small number, and
#    NOT proportional to the VRAM saved. This is a real, load-bearing
#    cost to plan capacity around, not a rough guess pulled from nowhere;
#    see the RAMLedger admission in swap_orchestrator.py, which will
#    correctly refuse to admit the Critic tier on a RAM-constrained host
#    rather than silently OOMing system memory instead of GPU memory.
#
# The Planner and Sketch tiers are small enough already (6.5GB + 3.0GB =
# 9.5GB combined) that no offload is applied to either even in low-VRAM
# mode — there's nothing to gain and every offload adds real latency.
# ---------------------------------------------------------------------------
LOW_VRAM_REGISTRY: dict[Tier, ModelSpec] = {
    Tier.PLANNER: REGISTRY[Tier.PLANNER],   # unchanged — already small
    Tier.SKETCH: REGISTRY[Tier.SKETCH],     # unchanged — already small
    Tier.POLISH_DEFAULT: REGISTRY[Tier.POLISH_DEFAULT],  # unchanged — 8.0GB already fits
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-Edit-2511",
        vram_gb=10.0, ram_gb=10.0,   # ESTIMATE — see mechanism (1) above;
                                      # validate against a real run and adjust
        quantization="NF4 + diffusers enable_model_cpu_offload()",
        fallback_tier=Tier.POLISH_DEFAULT,
    ),
    Tier.CRITIC: ModelSpec(
        tier=Tier.CRITIC, name="Gemma 4 31B Dense",
        vram_gb=12.0, ram_gb=40.0,   # ESTIMATE — see mechanism (2) above,
                                      # this one is a real, computed lower
                                      # bound from bnb's documented fp32
                                      # CPU-offload behavior, not a guess;
                                      # validate against a real run regardless
        quantization="NF4 (GPU-resident) + FP32 CPU offload via plain "
                      "transformers+bitsandbytes, NOT unsloth — see "
                      "critic_worker.py's _load() offload branch",
    ),
}


def get_registry(low_vram: bool = False) -> dict[Tier, ModelSpec]:
    """The single entry point for "which VRAM/RAM budget numbers are we
    operating under" — used by VRAMLedger/RAMLedger construction and by
    inference/factory.py when deciding which offload kwargs to pass each
    backend. Keeping this as one function (rather than scattering
    `if low_vram: ... else: ...` checks across the codebase) is what
    keeps the ledger, the orchestrator, and the backends from being able
    to silently disagree about which registry is active.
    """
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
    _load_attempts: int = field(default=0, init=False)
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
        await asyncio.sleep(0.01)
        base = {"tier": self.spec.tier.value, "mock_output": True, "input_echo": kwargs}

        # Tier-appropriate mock payloads so flows.py's field mapping (which
        # reads specific keys like vq_tokens_ref / critique_result) has
        # something real to work with end-to-end, without a GPU or real
        # weights. A real backend's run() is expected to return these same
        # keys — this is the de facto output contract each tier owes the
        # orchestration layer.
        if self.spec.tier == Tier.PLANNER:
            base["mock_output_text"] = "(mock planner reply)"
        elif self.spec.tier == Tier.SKETCH:
            rev = kwargs.get("planner_output", {}).get("input_echo", {})
            base["vq_tokens_ref"] = f"vq_grid::{hash(str(rev)) & 0xFFFF:x}"
            base["confidence_map_ref"] = f"conf_map::{hash(str(rev)) & 0xFFFF:x}"
        elif self.spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY):
            base["image_ref"] = f"render://mock/{self.spec.tier.value}/{hash(str(kwargs)) & 0xFFFF:x}.png"
            base["verifier_scores"] = {
                "clip_alignment": 0.9, "ocr_readability": 0.95,
                "layout_iou": 0.88, "aesthetic": 0.8, "handoff_consistency": 0.97,
            }
        elif self.spec.tier == Tier.CRITIC:
            base["critique_result"] = {
                "critique_source": "gemma4_31b_frozen",
                "overall_score": 0.82,
                "dimensions": {
                    "visual_hierarchy": {"score": 0.8, "note": "Clear primary action."},
                },
                "suggested_edits": [],
                "raw_model_output_ref": None,
            }
        return base


class TorchBackend(ModelBackend):
    """Real integration point for the inference layer. Left unimplemented
    on purpose — wire actual `transformers` / `diffusers` / Unsloth load
    calls here (see PRD §6 for the exact checkpoint + quantization per
    tier) when that layer is built. Deliberately fails loudly rather than
    pretending to work, so nobody accidentally ships this stub."""

    async def load(self) -> None:
        raise NotImplementedError(
            f"TorchBackend.load() for tier '{self.spec.tier.value}' "
            f"({self.spec.name}) is not implemented yet — this is the "
            "inference-pipeline layer, not part of the orchestrator build. "
            "Use MockBackend for orchestrator dev/testing."
        )

    async def unload(self) -> None:
        raise NotImplementedError

    async def run(self, **kwargs):
        raise NotImplementedError


def default_backend_factory(spec: ModelSpec) -> ModelBackend:
    """Swap this to TorchBackend once the inference layer exists. Random
    jitter on mock latency keeps tests honest about ordering assumptions."""
    return MockBackend(spec=spec, load_latency_s=0.01 + random.random() * 0.01)
