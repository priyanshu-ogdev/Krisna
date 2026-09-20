from __future__ import annotations

import pytest

from krisna_inference.orchestrator.exceptions import InvalidTransitionError
from krisna_inference.orchestrator.state_machine import ResidencyState, StateMachine, SwapTrigger


def test_initial_state_is_idle_resident():
    sm = StateMachine()
    assert sm.state == ResidencyState.IDLE_RESIDENT
    assert sm.is_conversation_eligible()
    assert not sm.is_in_flight()


def test_full_finalize_happy_path():
    sm = StateMachine()
    sm.apply(SwapTrigger.FINALIZE_REQUESTED)
    assert sm.state == ResidencyState.SWAPPING_TO_POLISH
    assert sm.is_in_flight()
    assert not sm.is_conversation_eligible()

    sm.apply(SwapTrigger.POLISH_LOAD_COMPLETE)
    assert sm.state == ResidencyState.POLISH_RESIDENT

    sm.apply(SwapTrigger.POLISH_DONE)
    assert sm.state == ResidencyState.SWAPPING_BACK_FROM_POLISH

    sm.apply(SwapTrigger.BASELINE_RESTORED)
    assert sm.state == ResidencyState.IDLE_RESIDENT
    assert sm.is_conversation_eligible()


def test_full_critique_happy_path():
    sm = StateMachine()
    sm.apply(SwapTrigger.CRITIQUE_REQUESTED)
    assert sm.state == ResidencyState.SWAPPING_TO_CRITIC
    sm.apply(SwapTrigger.CRITIC_LOAD_COMPLETE)
    assert sm.state == ResidencyState.CRITIC_RESIDENT
    sm.apply(SwapTrigger.CRITIC_DONE)
    assert sm.state == ResidencyState.SWAPPING_BACK_FROM_CRITIC
    sm.apply(SwapTrigger.BASELINE_RESTORED)
    assert sm.state == ResidencyState.IDLE_RESIDENT


@pytest.mark.parametrize(
    "start_trigger",
    [SwapTrigger.FINALIZE_REQUESTED, SwapTrigger.CRITIQUE_REQUESTED],
)
def test_oom_during_swap_routes_to_recovery_and_back(start_trigger):
    sm = StateMachine()
    sm.apply(start_trigger)
    sm.apply(SwapTrigger.OOM)
    assert sm.state == ResidencyState.ERROR_RECOVERY
    assert sm.is_in_flight()
    sm.apply(SwapTrigger.RECOVERY_COMPLETE)
    assert sm.state == ResidencyState.IDLE_RESIDENT


@pytest.mark.parametrize(
    ("load_trigger", "expected_resident"),
    [
        (SwapTrigger.POLISH_LOAD_COMPLETE, ResidencyState.POLISH_RESIDENT),
        (SwapTrigger.CRITIC_LOAD_COMPLETE, ResidencyState.CRITIC_RESIDENT),
    ],
)
def test_oom_during_resident_routes_to_recovery_and_back(load_trigger, expected_resident):
    sm = StateMachine()
    if expected_resident == ResidencyState.POLISH_RESIDENT:
        sm.apply(SwapTrigger.FINALIZE_REQUESTED)
    else:
        sm.apply(SwapTrigger.CRITIQUE_REQUESTED)
    sm.apply(load_trigger)
    assert sm.state == expected_resident

    sm.apply(SwapTrigger.OOM)
    assert sm.state == ResidencyState.ERROR_RECOVERY
    assert sm.is_in_flight()
    sm.apply(SwapTrigger.RECOVERY_COMPLETE)
    assert sm.state == ResidencyState.IDLE_RESIDENT


def test_illegal_transition_raises():
    sm = StateMachine()
    with pytest.raises(InvalidTransitionError):
        # Can't complete a load before requesting one.
        sm.apply(SwapTrigger.POLISH_LOAD_COMPLETE)


def test_illegal_transition_from_polish_resident():
    sm = StateMachine()
    sm.apply(SwapTrigger.FINALIZE_REQUESTED)
    sm.apply(SwapTrigger.POLISH_LOAD_COMPLETE)
    with pytest.raises(InvalidTransitionError):
        # Can't jump straight to critique while polish is resident —
        # single GPU, one swap at a time, no interleaving.
        sm.apply(SwapTrigger.CRITIQUE_REQUESTED)


def test_cannot_double_finalize_from_swapping_state():
    sm = StateMachine()
    sm.apply(SwapTrigger.FINALIZE_REQUESTED)
    with pytest.raises(InvalidTransitionError):
        sm.apply(SwapTrigger.FINALIZE_REQUESTED)


def test_can_apply_matches_apply_legality():
    sm = StateMachine()
    assert sm.can_apply(SwapTrigger.FINALIZE_REQUESTED)
    assert sm.can_apply(SwapTrigger.CRITIQUE_REQUESTED)
    assert not sm.can_apply(SwapTrigger.POLISH_DONE)


def test_history_records_every_transition():
    sm = StateMachine()
    sm.apply(SwapTrigger.FINALIZE_REQUESTED)
    sm.apply(SwapTrigger.POLISH_LOAD_COMPLETE)
    assert sm.history == [
        (ResidencyState.IDLE_RESIDENT, SwapTrigger.FINALIZE_REQUESTED, ResidencyState.SWAPPING_TO_POLISH),
        (ResidencyState.SWAPPING_TO_POLISH, SwapTrigger.POLISH_LOAD_COMPLETE, ResidencyState.POLISH_RESIDENT),
    ]
