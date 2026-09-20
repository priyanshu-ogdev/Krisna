"""Swap Orchestrator state machine — PRD §7.4.

The PRD names this section ("Formal Model Swap Orchestrator spec (§7.4):
state enum, transition table, load/unload sequencing, OOM handling") but the
provided document text was cut off before §7.4's body. This module is that
spec, reconstructed from what IS fully specified elsewhere in the document —
§5.3's three sequence diagrams, §3's goals ("Model transitions ... complete
without user-visible failure, even under memory pressure" and the 16-24GB
envelope), and §6's fallback_tier relationship (POLISH_QUALITY ->
POLISH_DEFAULT) — rather than invented from scratch. Treat this as the
concrete proposal for §7.4; if the original spec differs, the transition
table below is the part to reconcile against it.

ResidencyState is distinct from DesignState.stage (design_state.py):
ResidencyState is "what's physically loaded on the GPU right now" and is
shared across ALL sessions (single GPU, one orchestrator). DesignState.stage
is per-session product state. A single ResidencyState transition (e.g.
IDLE_RESIDENT -> SWAPPING_TO_POLISH) serves whichever session's finalize
request triggered it; other sessions' conversational turns are rejected
with SwapBusyError until the swap completes, per §5.3's "No swap occurs ...
this is the loop that runs dozens of times per session" framing — only ONE
loop (whichever session currently owns the swap lock) is blocked, but only
one swap can be in flight system-wide because there is one GPU.
"""

from __future__ import annotations

from enum import Enum


class ResidencyState(str, Enum):
    # Baseline: Planner + Sketch tier resident (always_resident=True tiers).
    # This is the ONLY state in which conversational turns can run.
    IDLE_RESIDENT = "idle_resident"

    # Transient: unloading Planner+Sketch, then loading the target polish
    # tier. No tier is guaranteed usable mid-transition.
    SWAPPING_TO_POLISH = "swapping_to_polish"
    POLISH_RESIDENT = "polish_resident"
    SWAPPING_BACK_FROM_POLISH = "swapping_back_from_polish"

    SWAPPING_TO_CRITIC = "swapping_to_critic"
    CRITIC_RESIDENT = "critic_resident"
    SWAPPING_BACK_FROM_CRITIC = "swapping_back_from_critic"

    # Entered only when OOM recovery (retry + fallback tier, see
    # swap_orchestrator.py) has been exhausted mid-swap. This state exists
    # so "no user-visible failure" is enforceable in code: nothing may
    # observe the orchestrator sitting here — recover() always drives it
    # back to IDLE_RESIDENT (restoring Planner+Sketch) before returning
    # control, even when the triggering request ultimately fails.
    ERROR_RECOVERY = "error_recovery"


class SwapTrigger(str, Enum):
    FINALIZE_REQUESTED = "finalize_requested"
    POLISH_LOAD_COMPLETE = "polish_load_complete"
    POLISH_DONE = "polish_done"
    BASELINE_RESTORED = "baseline_restored"

    CRITIQUE_REQUESTED = "critique_requested"
    CRITIC_LOAD_COMPLETE = "critic_load_complete"
    CRITIC_DONE = "critic_done"

    OOM = "oom"
    RECOVERY_COMPLETE = "recovery_complete"


# Transition table: (state, trigger) -> next_state.
# Any (state, trigger) pair not listed here is illegal — see
# StateMachine.apply(), which looks this table up and raises
# InvalidTransitionError on a miss rather than guessing.
TRANSITIONS: dict[tuple[ResidencyState, SwapTrigger], ResidencyState] = {
    (ResidencyState.IDLE_RESIDENT, SwapTrigger.FINALIZE_REQUESTED): ResidencyState.SWAPPING_TO_POLISH,
    (ResidencyState.SWAPPING_TO_POLISH, SwapTrigger.POLISH_LOAD_COMPLETE): ResidencyState.POLISH_RESIDENT,
    (ResidencyState.POLISH_RESIDENT, SwapTrigger.POLISH_DONE): ResidencyState.SWAPPING_BACK_FROM_POLISH,
    (ResidencyState.SWAPPING_BACK_FROM_POLISH, SwapTrigger.BASELINE_RESTORED): ResidencyState.IDLE_RESIDENT,

    (ResidencyState.IDLE_RESIDENT, SwapTrigger.CRITIQUE_REQUESTED): ResidencyState.SWAPPING_TO_CRITIC,
    (ResidencyState.SWAPPING_TO_CRITIC, SwapTrigger.CRITIC_LOAD_COMPLETE): ResidencyState.CRITIC_RESIDENT,
    (ResidencyState.CRITIC_RESIDENT, SwapTrigger.CRITIC_DONE): ResidencyState.SWAPPING_BACK_FROM_CRITIC,
    (ResidencyState.SWAPPING_BACK_FROM_CRITIC, SwapTrigger.BASELINE_RESTORED): ResidencyState.IDLE_RESIDENT,

    # OOM can strike during outbound swap or during runtime generation.
    # Both route to the same recovery state — recovery restores baseline residency.
    (ResidencyState.SWAPPING_TO_POLISH, SwapTrigger.OOM): ResidencyState.ERROR_RECOVERY,
    (ResidencyState.SWAPPING_TO_CRITIC, SwapTrigger.OOM): ResidencyState.ERROR_RECOVERY,
    (ResidencyState.POLISH_RESIDENT, SwapTrigger.OOM): ResidencyState.ERROR_RECOVERY,
    (ResidencyState.CRITIC_RESIDENT, SwapTrigger.OOM): ResidencyState.ERROR_RECOVERY,
    (ResidencyState.ERROR_RECOVERY, SwapTrigger.RECOVERY_COMPLETE): ResidencyState.IDLE_RESIDENT,
}

# States in which a conversational turn (§5.3 "Conversational turn") may
# run. Everywhere else, incoming turns must be rejected/queued — Planner
# is not resident.
CONVERSATION_ELIGIBLE_STATES = frozenset({ResidencyState.IDLE_RESIDENT})

# States that represent "a swap is in flight" — used to serialize access.
IN_FLIGHT_STATES = frozenset(
    {
        ResidencyState.SWAPPING_TO_POLISH,
        ResidencyState.SWAPPING_BACK_FROM_POLISH,
        ResidencyState.SWAPPING_TO_CRITIC,
        ResidencyState.SWAPPING_BACK_FROM_CRITIC,
        ResidencyState.ERROR_RECOVERY,
    }
)


class StateMachine:
    """Pure, side-effect-free transition enforcement. The orchestrator
    (swap_orchestrator.py) owns actually loading/unloading weights and
    calls this class to validate + record each step, so "is this legal"
    and "how do we make it happen" stay separate and each is separately
    testable."""

    def __init__(self, initial: ResidencyState = ResidencyState.IDLE_RESIDENT) -> None:
        self._state = initial
        self.history: list[tuple[ResidencyState, SwapTrigger, ResidencyState]] = []

    @property
    def state(self) -> ResidencyState:
        return self._state

    def can_apply(self, trigger: SwapTrigger) -> bool:
        return (self._state, trigger) in TRANSITIONS

    def apply(self, trigger: SwapTrigger) -> ResidencyState:
        key = (self._state, trigger)
        if key not in TRANSITIONS:
            from krisna_inference.orchestrator.exceptions import InvalidTransitionError

            raise InvalidTransitionError(self._state.value, trigger.value)
        next_state = TRANSITIONS[key]
        self.history.append((self._state, trigger, next_state))
        self._state = next_state
        return next_state

    def is_conversation_eligible(self) -> bool:
        return self._state in CONVERSATION_ELIGIBLE_STATES

    def is_in_flight(self) -> bool:
        return self._state in IN_FLIGHT_STATES
