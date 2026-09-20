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
        # bf16, ~14GB ESTIMATE (not yet measured on real hardware) —
        # corrected from a previous 8.0GB figure that assumed NF4
        # quantization this backend never actually applies (confirmed by
        # reading polish_default_backend.py: `torch_dtype=resolve_dtype
        # (self.dtype)` with no `quantization_config`, and deliberately
        # so — see that file's `load()` comment on why NF4 here would be
        # a train/inference precision mismatch against the bf16/fp16-
        # trained LoRA). Arithmetic: Z-Image-Turbo is 6B params (Tongyi-
        # MAI, 2025) at 2 bytes/param bf16 = ~12GB for the DiT alone,
        # plus a text encoder and VAE the model card doesn't break out
        # separately — 14.0 is a conservative round-up, consistent with
        # Tongyi-MAI's own published "<16GB" full-pipeline guidance as an
        # upper bound. Needs a real hardware measurement to replace this
        # estimate with a confirmed number — see
        # docs/review/13_ram_offload_and_precision_audit.md.
        tier=Tier.POLISH_DEFAULT, name="Z-Image-Turbo", vram_gb=14.0,
        quantization="bf16 (deliberately unquantized — matches LoRA training precision)",
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
# THREE real offload mechanisms are in play here (was two — Polish
# Default's is new, added alongside the vram_gb correction above), with
# very different RAM-cost implications — conflating them would give a
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
#    Polish Default is bf16, not NF4 (see REGISTRY entry above) — the
#    GH #10800 incompatibility is specific to bnb 4-bit tensors and does
#    not apply to this tier, but `enable_model_cpu_offload()` is used
#    here regardless, for consistency with Polish Quality's mechanism
#    rather than introducing a second, differently-justified offload
#    path for one tier only (see polish_default_backend.py's comment).
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
    Tier.POLISH_DEFAULT: ModelSpec(
        tier=Tier.POLISH_DEFAULT, name="Z-Image-Turbo",
        # ESTIMATE, same caveat as the REGISTRY entry above — this one
        # additionally assumes `enable_model_cpu_offload()`'s real-world
        # saving here is bounded by which single submodule dominates this
        # model's size. Z-Image-Turbo's DiT transformer (the bulk of its
        # 6B params) can't itself be offloaded away mid-forward-pass —
        # only the text encoder and VAE can sit on CPU while idle. So the
        # GPU floor is roughly "DiT alone" (~12GB) rather than "DiT +
        # everything else" (~14GB): a real but modest ~2GB saving, unlike
        # Polish Quality's offload (mechanism 1 above, same call) where
        # more of the pipeline's weight is outside its single largest
        # submodule. Fits its own 12GB low-VRAM envelope with
        # approximately zero headroom. UPGRADE: this comment used to say
        # "same situation as the Critic entry below" — that cross-
        # reference went stale once Critic's own zero-headroom sizing was
        # fixed (11.5/45.0, below) and this comment wasn't updated
        # alongside it. Genuinely different situations, not just an
        # unfixed duplicate: Critic's GPU-resident target was a
        # continuously adjustable knob (how much of an LLM stays
        # resident vs. offloads, with a computable RAM cost per GB
        # moved), so shading it down and recomputing the RAM side was a
        # real, justified fix. This tier's 12GB figure is already "the
        # DiT alone, everything else already offloaded" — there's no
        # further partial-offload knob to turn without switching to
        # `enable_sequential_cpu_offload()`'s per-layer latency cost,
        # which this project has declined elsewhere for the same reason.
        # See swap_orchestrator.py's `vram_safety_margin_gb` for the
        # opt-in mitigation, and validate this split against a real run
        # before trusting it at the boundary.
        vram_gb=12.0, ram_gb=2.0,
        quantization="bf16 (deliberately unquantized) + diffusers enable_model_cpu_offload()",
    ),
    Tier.POLISH_QUALITY: ModelSpec(
        tier=Tier.POLISH_QUALITY, name="Qwen-Image-Edit-2511",
        vram_gb=10.0, ram_gb=10.0,   # ESTIMATE — see mechanism (1) above;
                                      # validate against a real run and adjust
        quantization="NF4 + diffusers enable_model_cpu_offload()",
        fallback_tier=Tier.POLISH_DEFAULT,
    ),
    Tier.CRITIC: ModelSpec(
        tier=Tier.CRITIC, name="Gemma 4 31B Dense",
        vram_gb=11.5, ram_gb=45.0,   # UPGRADE (was vram_gb=12.0, ram_gb=40.0):
                                      # 12.0 was picked to exactly equal the
                                      # 12GB low-VRAM target envelope
                                      # (service.py's KRISNA_VRAM_ENVELOPE_GB
                                      # default) — zero headroom on paper
                                      # against a target that itself doesn't
                                      # account for CUDA context, allocator
                                      # fragmentation, or activation memory.
                                      # Dropping the GPU-resident target to
                                      # 11.5GB bakes in ~0.5GB (~4%) real
                                      # headroom against a 12GB card. This is
                                      # NOT free — offloading less to the GPU
                                      # means MORE params move to CPU at FP32
                                      # (see mechanism (2) above), so ram_gb
                                      # must go up too, not stay at 40.0:
                                      # recomputed via the same method as
                                      # before (bytes/param implied by the
                                      # 18.0GB full-NF4 footprint above,
                                      # applied to the new 11.5GB GPU split)
                                      # gives ~44.3GB, rounded up to 45.0GB —
                                      # still comfortably under the 48.0GB
                                      # default RAM envelope (swap_orchestrator's
                                      # ram_envelope_gb), so this doesn't just
                                      # relocate the zero-headroom problem from
                                      # the VRAM ledger onto the RAM ledger.
                                      # KRISNA_CRITIC_MAX_GPU_GB's default in
                                      # backends/factory.py MUST stay in sync
                                      # with this vram_gb value.
                                      # REGRESSION NOTE (now recurring —
                                      # third time this fix has been lost
                                      # from an uploaded working copy in
                                      # this review; see docs/review's
                                      # latest phase for the full pattern):
                                      # if this drifts back to 12.0/40.0
                                      # again, the issue is upstream of any
                                      # single fix — something in how these
                                      # working copies get produced between
                                      # sessions isn't carrying edits forward.
                                      # Still an estimate pending a real run.
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
            base["design_state_delta"] = {
                "stage": "sketching",
                "constraint_updates": kwargs.get("constraints") or {},
                "tool_call": None,
                "reasoning_note": "Proceeding with current constraints.",
            }
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
