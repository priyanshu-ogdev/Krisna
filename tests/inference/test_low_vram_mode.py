from __future__ import annotations

import pytest

from krisna_inference.orchestrator.model_registry import (
    HIGH_VRAM_REGISTRY,
    LOW_VRAM_REGISTRY,
    REGISTRY,
    MockBackend,
    ModelSpec,
    Tier,
    get_registry,
)
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator


def fast_mock_factory(spec: ModelSpec) -> MockBackend:
    return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)


class TestGetRegistry:
    def test_default_returns_full_vram_registry(self):
        assert get_registry() is REGISTRY

    def test_low_vram_true_returns_low_vram_registry(self):
        assert get_registry(low_vram=True) is LOW_VRAM_REGISTRY

    def test_high_vram_true_returns_high_vram_registry(self):
        assert get_registry(high_vram=True) is HIGH_VRAM_REGISTRY

    def test_low_and_high_vram_both_true_raises(self):
        with pytest.raises(ValueError):
            get_registry(low_vram=True, high_vram=True)

    def test_full_registry_has_zero_ram_gb_everywhere(self):
        for spec in REGISTRY.values():
            assert spec.ram_gb == 0.0

    def test_high_vram_registry_has_zero_ram_gb_everywhere(self):
        for spec in HIGH_VRAM_REGISTRY.values():
            assert spec.ram_gb == 0.0

    def test_high_vram_registry_has_all_tiers_always_resident(self):
        for spec in HIGH_VRAM_REGISTRY.values():
            assert spec.always_resident is True

    def test_low_vram_registry_reduces_the_offloadable_tiers(self):
        assert LOW_VRAM_REGISTRY[Tier.POLISH_DEFAULT].vram_gb < REGISTRY[Tier.POLISH_DEFAULT].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.POLISH_QUALITY].vram_gb < REGISTRY[Tier.POLISH_QUALITY].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.POLISH_DEFAULT].ram_gb > 0.0
        assert LOW_VRAM_REGISTRY[Tier.POLISH_QUALITY].ram_gb > 0.0

    def test_low_vram_registry_leaves_small_tiers_unchanged(self):
        assert LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb == REGISTRY[Tier.PLANNER].vram_gb
        assert LOW_VRAM_REGISTRY[Tier.PLANNER].ram_gb == 0.0
        assert LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb == REGISTRY[Tier.SKETCH].vram_gb

    def test_every_swappable_tier_fits_a_12gb_envelope_alone_in_low_vram_mode(self):
        baseline_vram = LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb + LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb
        for tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY):
            assert LOW_VRAM_REGISTRY[tier].vram_gb <= 12.0
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
            assert orch.ram_ledger.used_gb == 0.0
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_finalize_quality_succeeds_and_cleans_up_both_ledgers(self):
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
            assert result.tier_used == Tier.PLANNER
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_ram_budget_too_small_fails_cleanly_even_if_vram_fits(self):
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=16.0, ram_envelope_gb=5.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            result = await orch.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
            assert result.ok is False
            assert orch.conversation_available()
        finally:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_low_vram_mode_still_enforces_mandatory_exclusivity(self):
        orch = SwapOrchestrator(
            low_vram=True, envelope_gb=12.0, ram_envelope_gb=48.0,
            backend_factory=fast_mock_factory,
        )
        await orch.start()
        try:
            await orch.request_finalize(preferred_tier=Tier.POLISH_DEFAULT)
        finally:
            await orch.shutdown()
