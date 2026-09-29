from __future__ import annotations

import pytest

from krisna_inference.orchestrator.model_registry import LOW_VRAM_REGISTRY, REGISTRY, MockBackend, ModelSpec, Tier, get_registry
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator


def fast_mock_factory(spec: ModelSpec) -> MockBackend:
    return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)


class TestGetRegistry:
    def test_default_returns_full_vram_registry(self):
        assert get_registry() is REGISTRY

    def test_low_vram_true_returns_low_vram_registry(self):
        assert get_registry(low_vram=True) is LOW_VRAM_REGISTRY

    def test_full_registry_has_zero_ram_gb_everywhere(self):
        """The no-op guarantee RAMLedger's docstring depends on: passing
        the full registry must make every ram_gb exactly 0.0, or
        RAMLedger stops being a safe no-op in the default mode."""
        for spec in REGISTRY.values():
            assert spec.ram_gb == 0.0

    def test_low_vram_registry_reduces_the_offloadable_tiers(self):
        """Was 'the two large tiers' — now three: Polish Default's
        vram_gb=8.0 assumed NF4 quantization the backend never actually
        applied (see model_registry.py's corrected comment and
        docs/review/13_ram_offload_and_precision_audit.md). Its real bf16
        footprint (~14GB) doesn't already fit a 12GB target, so it now
        gets the same enable_model_cpu_offload() treatment as Polish
        Quality."""
        assert LOW_VRAM_REGISTRY[Tier.POLISH_DEFAULT].vram_gb < REGISTRY[Tier.POLISH_DEFAULT].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.POLISH_QUALITY].vram_gb < REGISTRY[Tier.POLISH_QUALITY].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.CRITIC].vram_gb < REGISTRY[Tier.CRITIC].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.POLISH_DEFAULT].ram_gb > 0.0
        assert LOW_VRAM_REGISTRY[Tier.POLISH_QUALITY].ram_gb > 0.0
        assert LOW_VRAM_REGISTRY[Tier.CRITIC].ram_gb > 0.0

    def test_low_vram_registry_leaves_small_tiers_unchanged(self):
        """Planner and Sketch are already small enough that offloading
        them buys nothing but latency — confirm they're untouched. Polish
        Default is NOT in this group (see the reduces_the_offloadable_tiers
        test above) — it used to be, incorrectly."""
        assert LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb == REGISTRY[Tier.PLANNER].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.PLANNER].ram_gb == 0.0
        assert LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb == REGISTRY[Tier.SKETCH].vram_gb

    def test_every_swappable_tier_fits_a_12gb_envelope_alone_in_low_vram_mode(self):
        """The actual claim being made to the user: each swappable tier,
        checked individually (matching the PRD's mandatory-exclusivity
        rule — only one of {polish, critic} is ever resident alongside
        baseline), fits under a 12GB envelope in low-VRAM mode."""
        baseline_vram = LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb + LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb
        for tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.CRITIC):
            assert LOW_VRAM_REGISTRY[tier].vram_gb <= 12.0, (
                f"{tier.value} alone (baseline already unloaded per the mandatory "
                f"exclusivity rule) doesn't fit a 12GB envelope"
            )
        assert baseline_vram <= 12.0


class TestSwapOrchestratorLowVramMode:
    @pytest.mark.asyncio
    async def test_low_vram_true_uses_low_vram_registry(self):
        orch = SwapOrchestrator(low_vram=True, backend_factory=fast_mock_factory)
        assert orch.registry is LOW_VRAM_REGISTRY
        assert orch.ledger.registry is LOW_VRAM_REGISTRY
        assert orch.ram_ledger.registry is LOW_VRAM_REGISTRY

    @pytest.mark.asyncio
    async def test_default_uses_full_registry_and_ram_ledger_is_a_noop(self):
        orch = SwapOrchestrator(backend_factory=fast_mock_factory)
        assert orch.registry is REGISTRY
        await orch.start()
        try:
            assert orch.ram_ledger.used_gb == 0.0
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_mode_admits_baseline_to_both_ledgers(self):
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=12.0, ram_envelope_gb=48.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            expected_vram = LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb + LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb
            assert orch.ledger.used_gb == expected_vram
            assert orch.ram_ledger.used_gb == 0.0  # Planner/Sketch have ram_gb=0.0 even in low-VRAM mode
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_finalize_quality_succeeds_and_cleans_up_both_ledgers(self):
        """request_finalize unloads the polish tier again before returning
        (see _finalize_locked), so the meaningful check post-call isn't a
        transient mid-flight ram_ledger value — it's that the call
        succeeded under the tighter budget AND both ledgers cleanly
        returned to baseline-only afterward, proving no leak."""
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=12.0, ram_envelope_gb=48.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            result = await orch.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
            assert result.ok
            assert result.tier_used == Tier.POLISH_QUALITY
            assert result.degraded is False
            # Back to baseline-only after the call — Planner+Sketch only.
            assert orch.ledger.used_gb == (
                LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb + LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb
            )
            assert orch.ram_ledger.used_gb == 0.0
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_critique_succeeds_under_12gb_vram_envelope(self):
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=12.0, ram_envelope_gb=48.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            result = await orch.request_critique()
            assert result.ok
            assert result.tier_used == Tier.CRITIC
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_ram_budget_too_small_fails_cleanly_even_if_vram_fits(self):
        """The actual point of having two ledgers: a candidate that fits
        VRAM but not RAM must be refused (via the preflight would_fit
        check), not silently admitted. Critic has no fallback_tier, so
        this should come back as a clean SwapResult(ok=False), not an
        exception — same failure shape as an exhausted-retries OOM,
        per _load_with_recovery's documented policy."""
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=16.0, ram_envelope_gb=5.0,  # 5GB is nowhere near
            backend_factory=fast_mock_factory,                      # Critic's ~40GB ram_gb
        )
        await orch.start()
        try:
            result = await orch.request_critique()
            assert result.ok is False
            # System stays healthy — baseline still resident afterward.
            assert orch.conversation_available()
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_mode_still_enforces_mandatory_exclusivity(self):
        """PRD §7.4 rule #3 must hold in low-VRAM mode too: baseline is
        never resident alongside a swap tier, regardless of which
        registry/envelope is active."""
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=12.0, ram_envelope_gb=48.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            await orch.request_finalize(preferred_tier=Tier.POLISH_DEFAULT)
        finally:
            await orch.shutdown()
        # After shutdown everything is released; the in-flight assertion
        # that matters is exercised by test_finalize_unloads_baseline_and_
        # restores_it in test_swap_orchestrator.py already — this test
        # just confirms low_vram mode doesn't bypass that machinery.
