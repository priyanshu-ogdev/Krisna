from __future__ import annotations

import asyncio

import pytest

from krisna_inference.orchestrator.exceptions import OOMRecoveryExhausted, SwapBusyError
from krisna_inference.orchestrator.model_registry import MockBackend, ModelSpec, Tier
from krisna_inference.orchestrator.state_machine import ResidencyState
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator


@pytest.mark.asyncio
async def test_start_loads_baseline(started_orchestrator: SwapOrchestrator):
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.state.state == ResidencyState.IDLE_RESIDENT
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_conversational_turn_happy_path(started_orchestrator: SwapOrchestrator):
    out = await started_orchestrator.run_conversational_turn(message="hi")
    assert out["planner"]["mock_output"] is True
    assert out["sketch"]["mock_output"] is True


@pytest.mark.asyncio
async def test_finalize_unloads_baseline_and_restores_it(started_orchestrator: SwapOrchestrator):
    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_DEFAULT)
    assert result.ok
    assert result.tier_used == Tier.POLISH_DEFAULT
    assert not result.degraded
    # Baseline must be restored afterward.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert not started_orchestrator.backends[Tier.POLISH_DEFAULT].is_loaded
    assert started_orchestrator.state.state == ResidencyState.IDLE_RESIDENT


@pytest.mark.asyncio
async def test_critique_unloads_baseline_and_restores_it(started_orchestrator: SwapOrchestrator):
    result = await started_orchestrator.request_critique()
    assert result.ok
    assert result.tier_used == Tier.CRITIC
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert not started_orchestrator.backends[Tier.CRITIC].is_loaded
    assert started_orchestrator.state.state == ResidencyState.IDLE_RESIDENT


@pytest.mark.asyncio
async def test_conversation_rejected_while_swap_in_flight(started_orchestrator: SwapOrchestrator):
    # Slow down the polish load so we can observe the in-flight state.
    started_orchestrator.backends[Tier.POLISH_DEFAULT].load_latency_s = 0.05

    finalize_task = asyncio.create_task(
        started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_DEFAULT)
    )
    await asyncio.sleep(0.01)  # let the swap actually start
    assert not started_orchestrator.conversation_available()
    with pytest.raises(SwapBusyError):
        await started_orchestrator.run_conversational_turn(message="hi")

    result = await finalize_task
    assert result.ok
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_second_swap_rejected_while_first_in_flight(started_orchestrator: SwapOrchestrator):
    started_orchestrator.backends[Tier.POLISH_DEFAULT].load_latency_s = 0.05
    t1 = asyncio.create_task(started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_DEFAULT))
    await asyncio.sleep(0.01)
    with pytest.raises(SwapBusyError):
        await started_orchestrator.request_critique()
    await t1


@pytest.mark.asyncio
async def test_oom_falls_back_to_polish_default(started_orchestrator: SwapOrchestrator):
    # Force POLISH_QUALITY to always OOM; it has a configured fallback
    # (POLISH_DEFAULT per §6's stack table).
    started_orchestrator.backends[Tier.POLISH_QUALITY].fail_loads = 999

    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
    assert result.ok
    assert result.tier_used == Tier.POLISH_DEFAULT
    assert result.degraded is True
    # System healthy afterward.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded


@pytest.mark.asyncio
async def test_oom_recovers_after_transient_failure_within_retry_budget(
    started_orchestrator: SwapOrchestrator,
):
    # Fails once, then succeeds — within max_retries (default 2).
    started_orchestrator.backends[Tier.POLISH_DEFAULT].fail_loads = 1
    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_DEFAULT)
    assert result.ok
    assert result.tier_used == Tier.POLISH_DEFAULT
    assert not result.degraded


@pytest.mark.asyncio
async def test_critic_oom_has_no_fallback_and_restores_baseline(
    started_orchestrator: SwapOrchestrator,
):
    started_orchestrator.backends[Tier.CRITIC].fail_loads = 999
    result = await started_orchestrator.request_critique()
    assert not result.ok
    assert result.error is not None
    assert "critic" in result.error
    # No user-visible failure of the SYSTEM: baseline must be back.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.state.state == ResidencyState.IDLE_RESIDENT
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_both_quality_and_fallback_oom_exhausts_recovery(
    started_orchestrator: SwapOrchestrator,
):
    started_orchestrator.backends[Tier.POLISH_QUALITY].fail_loads = 999
    started_orchestrator.backends[Tier.POLISH_DEFAULT].fail_loads = 999
    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
    assert not result.ok
    # System still healthy — this is the whole point of the recovery path.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_vram_budget_preflight_skips_straight_to_fallback(started_orchestrator: SwapOrchestrator):
    # Tiny envelope: POLISH_QUALITY (16GB) alone doesn't fit, but
    # POLISH_DEFAULT (14GB, corrected from an earlier 8GB estimate that
    # wrongly assumed NF4 quantization — see model_registry.py and
    # docs/review/13_ram_offload_and_precision_audit.md) does. Baseline
    # gets unloaded before the polish load in the real sequence, so
    # budget is checked against JUST the candidate tier size.
    orchestrator = started_orchestrator
    orchestrator.ledger.envelope_gb = 15.0  # smaller than POLISH_QUALITY's 16GB alone, bigger than POLISH_DEFAULT's 14GB
    result = await orchestrator.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
    assert result.ok
    assert result.tier_used == Tier.POLISH_DEFAULT  # 14GB fits under 15GB envelope
    assert result.degraded is True


@pytest.mark.asyncio
async def test_new_orchestrator_conversation_unavailable_before_start():
    orchestrator = SwapOrchestrator(
        backend_factory=lambda spec: MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)
    )
    # Baseline never loaded — is_conversation_eligible() is true (state
    # machine sits at IDLE_RESIDENT initially) but backends aren't actually
    # loaded. run_conversational_turn should surface that via the backend's
    # own "not loaded" error, not silently succeed.
    with pytest.raises(RuntimeError):
        await orchestrator.run_conversational_turn(message="hi")


@pytest.mark.asyncio
async def test_run_oom_falls_back_to_polish_default(started_orchestrator: SwapOrchestrator):
    # Simulate runtime OOM during POLISH_QUALITY generation (run()).
    # It has a configured fallback (POLISH_DEFAULT).
    started_orchestrator.backends[Tier.POLISH_QUALITY].fail_runs = 999

    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
    assert result.ok
    assert result.tier_used == Tier.POLISH_DEFAULT
    assert result.degraded is True
    # System healthy and baseline restored afterward.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_run_oom_both_quality_and_fallback_exhausts_recovery(started_orchestrator: SwapOrchestrator):
    # Simulate runtime OOM on both POLISH_QUALITY and POLISH_DEFAULT.
    started_orchestrator.backends[Tier.POLISH_QUALITY].fail_runs = 999
    started_orchestrator.backends[Tier.POLISH_DEFAULT].fail_runs = 999

    result = await started_orchestrator.request_finalize(preferred_tier=Tier.POLISH_QUALITY)
    assert not result.ok
    # System healthy and baseline restored afterward.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_critic_run_oom_restores_baseline(started_orchestrator: SwapOrchestrator):
    # Simulate runtime OOM during CRITIC generation (run()).
    started_orchestrator.backends[Tier.CRITIC].fail_runs = 999

    result = await started_orchestrator.request_critique()
    assert not result.ok
    assert result.error is not None
    assert "critic" in result.error
    # Baseline restored.
    assert started_orchestrator.backends[Tier.PLANNER].is_loaded
    assert started_orchestrator.backends[Tier.SKETCH].is_loaded
    assert started_orchestrator.conversation_available()

