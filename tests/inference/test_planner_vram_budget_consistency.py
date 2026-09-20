from __future__ import annotations

from krisna_inference.backends.planner_backend import PlannerBackend
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


class TestPlannerQuantizationMatchesDeclaredBudget:
    """Regression guard for a real bug: PlannerBackend used to default to
    dtype="bfloat16" with no quantization at all — ~18GB for a 9B model,
    nearly 3x model_registry.py's declared vram_gb=6.5 (sized for 4-bit)
    for this always_resident=True tier. That would have blown PRD §5.3's
    "Tier A must never exceed ~10GB resident" rule before the sketch tier
    or anything else could load in the idle/conversing state — silently,
    since nothing connects the backend's actual load() behavior to the
    registry's declared budget at runtime.
    """

    def test_planner_backend_quantizes_by_default(self):
        backend = PlannerBackend(REGISTRY[Tier.PLANNER])
        assert backend.quantize is True

    def test_planner_is_always_resident_in_registry(self):
        """The reason the quantization default matters at all: this tier
        stays loaded through the entire idle/conversing loop, not just
        transiently — see swap_orchestrator.py."""
        assert REGISTRY[Tier.PLANNER].always_resident is True

    def test_declared_budget_is_sized_for_4bit_not_bf16(self):
        """6.5GB is a 4-bit-scale number for a 9B model — a BF16 load
        (~18GB) would not fit this budget. This test exists so that if
        someone changes vram_gb here without checking the backend's
        actual dtype, or vice versa, the mismatch is at least documented
        by an explicit, named assertion rather than silent drift."""
        spec = REGISTRY[Tier.PLANNER]
        assert spec.vram_gb < 10.0, (
            "Planner's declared vram_gb looks sized for BF16, not 4-bit — "
            "if this was raised intentionally (e.g. after an eval found 4-bit "
            "inference quality unacceptable per PRD §15), also flip "
            "PlannerBackend's quantize default to False and re-check the "
            "full idle-state VRAM sum against vram_budget.py's 24GB envelope."
        )

    def test_idle_state_vram_sum_fits_declared_envelope(self):
        """The actual PRD §5.3 claim, checked directly: Planner (always
        resident) + Sketch (always resident, implied by the conversational
        loop never swapping it) must leave meaningful headroom under the
        24GB envelope for a Finalize or Critique swap to ever succeed."""
        from krisna_inference.orchestrator.vram_budget import VRAMLedger

        ledger = VRAMLedger()
        ledger.admit(Tier.PLANNER)
        ledger.admit(Tier.SKETCH)
        assert ledger.used_gb <= 10.0, (
            f"idle-state resident footprint is {ledger.used_gb}GB — PRD §5.3 "
            f"requires Tier A (planner+sketch) stay under ~10GB"
        )
        assert ledger.free_gb >= 12.0, (
            "not enough headroom left for a Finalize/Critique swap to fit "
            "under the 24GB envelope"
        )
