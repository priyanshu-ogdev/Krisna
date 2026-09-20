"""VRAM (and system-RAM) budget ledgers.

PRD §3 goal: "Every resident inference state fits inside a 16-24GB VRAM
envelope, even though the training/batch host is the same 48GB card."

This tracks declared-size accounting (ModelSpec.vram_gb) against a
configurable envelope, independent of whether a real GPU is present —
that's what makes it testable in CI. `probe_real_vram()` is an optional
supplement that queries torch.cuda if available, purely for logging/
diagnostics; the orchestrator's admission decisions are always based on
the declared ledger, not the live probe, so behavior is deterministic and
doesn't depend on what else happens to be running on the card.

Low-VRAM / CPU-offload mode (new): model_registry.get_registry(low_vram=True)
returns ModelSpecs with a lower `vram_gb` (the GPU-resident footprint under
CPU offload) and a populated `ram_gb` (the offloaded portion, now living in
system RAM instead). VRAMLedger and RAMLedger are both registry-driven
(constructed with `registry=...`) so the SAME ledger logic enforces either
budget correctly — passing the full-VRAM registry makes RAMLedger a no-op
(every ram_gb is 0.0) automatically, without a separate enabled/disabled
flag anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krisna_inference.orchestrator.exceptions import VRAMBudgetExceededError
from krisna_inference.orchestrator.model_registry import REGISTRY, ModelSpec, Tier


@dataclass
class _BaseLedger:
    """Shared accounting logic for VRAMLedger and RAMLedger — same
    admit/release/would_fit/snapshot shape, differing only in which
    ModelSpec attribute they sum (vram_gb vs ram_gb). Kept as a real base
    class rather than two independently-copy-pasted implementations, so
    the two can't silently drift apart — a fix here fixes both ledgers
    at once.
    """

    envelope_gb: float
    registry: dict = field(default_factory=lambda: REGISTRY)
    _resident: dict = field(default_factory=dict)
    _attr: str = "vram_gb"
    safety_margin_gb: float = 0.0
    """Extra headroom subtracted from the effective envelope before an
    admission check. Defaults to 0.0 for both ledgers — no behavior
    change unless a caller opts in. Left as an explicit opt-in rather
    than a nonzero default because at least one registered tier (Critic,
    low-VRAM mode) is sized to fit its envelope with exactly zero
    headroom on paper; a nonzero default here would make that tier
    permanently inadmissible (it has no fallback_tier), trading a real
    problem for a worse one. See swap_orchestrator.py's
    `vram_safety_margin_gb` for the caller-facing opt-in and the same
    reasoning in full. Real GPU memory has overhead (CUDA context,
    fragmentation, activations) the declared vram_gb estimate doesn't
    capture, so operators running close to the edge on real hardware
    should raise this explicitly rather than assume it's safe by
    default."""

    @property
    def resident_tiers(self) -> list:
        return list(self._resident.keys())

    @property
    def used_gb(self) -> float:
        return sum(getattr(spec, self._attr) for spec in self._resident.values())

    @property
    def free_gb(self) -> float:
        return self.envelope_gb - self.safety_margin_gb - self.used_gb

    def would_fit(self, tier) -> bool:
        spec = self.registry[tier]
        if tier in self._resident:
            return True
        return self.used_gb + getattr(spec, self._attr) <= self.envelope_gb - self.safety_margin_gb

    def admit(self, tier) -> None:
        """Reserve budget for a tier that is about to be loaded. Raises if
        it doesn't fit — callers should check would_fit() first, or catch
        this and fall back."""
        spec = self.registry[tier]
        if tier in self._resident:
            return
        cost = getattr(spec, self._attr)
        effective_envelope = self.envelope_gb - self.safety_margin_gb
        if self.used_gb + cost > effective_envelope:
            raise VRAMBudgetExceededError(
                requested_gb=self.used_gb + cost,
                budget_gb=self.envelope_gb,
                resident=[t.value for t in self._resident],
            )
        self._resident[tier] = spec

    def release(self, tier) -> None:
        self._resident.pop(tier, None)

    def snapshot(self) -> dict:
        return {
            "envelope_gb": self.envelope_gb,
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "resident": sorted(t.value for t in self._resident),
        }


@dataclass
class VRAMLedger(_BaseLedger):
    envelope_gb: float = 24.0     # upper bound of the §3 16-24GB envelope.
                                   # For the 12-16GB low-VRAM/CPU-offload
                                   # operating point, construct with
                                   # envelope_gb=16.0 (or 12.0) AND
                                   # registry=get_registry(low_vram=True) —
                                   # the two must be changed together, since
                                   # the full-VRAM registry's vram_gb values
                                   # won't fit a lowered envelope at all.
    card_total_gb: float = 48.0   # informational only — not the enforced budget
    _attr: str = field(default="vram_gb", repr=False)


@dataclass
class RAMLedger(_BaseLedger):
    """System-RAM counterpart to VRAMLedger — tracks the CPU-resident
    portion of any model loaded in low-VRAM/CPU-offload mode (see
    model_registry.py's ModelSpec.ram_gb). swap_orchestrator.py admits to
    both ledgers before a load and only proceeds if both succeed — a tier
    can fit its VRAM budget but blow the RAM budget (or vice versa on a
    VRAM-rich, RAM-poor box), and callers need to check both
    independently.
    """

    envelope_gb: float = 48.0     # default: assume a reasonably RAM-rich host;
                                   # override to your real system RAM, minus
                                   # headroom for the OS/other processes
    _attr: str = field(default="ram_gb", repr=False)


def probe_real_vram() -> dict | None:
    """Best-effort live GPU query, for diagnostics/logging only. Returns
    None if torch/CUDA isn't available — never raises, never used for
    admission decisions."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_gb": round(free_bytes / (1024**3), 2),
            "total_gb": round(total_bytes / (1024**3), 2),
        }
    except Exception:
        return None


def probe_real_ram() -> dict | None:
    """Best-effort live system-RAM query, for diagnostics/logging only —
    the RAMLedger counterpart to probe_real_vram(). Reads /proc/meminfo
    directly (no new hard dependency); returns None on non-Linux rather
    than adding a psutil dependency just for a diagnostics-only
    nice-to-have.
    """
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                key, _, rest = line.partition(":")
                meminfo[key] = rest.strip()
        total_kb = int(meminfo["MemTotal"].split()[0])
        avail_kb = int(meminfo.get("MemAvailable", meminfo["MemFree"]).split()[0])
        return {
            "free_gb": round(avail_kb / (1024**2), 2),
            "total_gb": round(total_kb / (1024**2), 2),
        }
    except Exception:
        return None
