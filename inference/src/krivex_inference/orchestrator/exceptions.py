"""Exceptions raised by the swap orchestrator and design state manager."""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for all orchestrator errors."""


class InvalidTransitionError(OrchestratorError):
    """Raised when a residency or session-stage transition is not legal
    from the current state."""

    def __init__(self, current_state: str, attempted: str) -> None:
        self.current_state = current_state
        self.attempted = attempted
        super().__init__(
            f"Cannot transition from '{current_state}' via '{attempted}' — "
            "not a legal edge in the state machine."
        )


class SwapBusyError(OrchestratorError):
    """Raised when a caller tries to start a conversational turn or a new
    swap while another swap is already in flight. The single-GPU design
    means only one swap operation can run at a time — this is not a bug,
    it's the whole point of the swap_lock."""


class SwapTimeoutError(OrchestratorError):
    """A load or unload operation exceeded its configured timeout."""

    def __init__(self, tier: str, operation: str, timeout_s: float) -> None:
        self.tier = tier
        self.operation = operation
        self.timeout_s = timeout_s
        super().__init__(f"{operation} of tier '{tier}' exceeded {timeout_s}s timeout")


class VRAMBudgetExceededError(OrchestratorError):
    """Raised when a proposed residency set would exceed the configured
    VRAM envelope, and no fallback tier is available to relieve it."""

    def __init__(self, requested_gb: float, budget_gb: float, resident: list[str]) -> None:
        self.requested_gb = requested_gb
        self.budget_gb = budget_gb
        self.resident = resident
        super().__init__(
            f"Requested residency needs {requested_gb:.1f}GB, budget is "
            f"{budget_gb:.1f}GB. Currently resident: {resident}"
        )


class OOMRecoveryExhausted(OrchestratorError):
    """Raised when a tier load hit OOM, all configured recovery steps
    (cache clear, fallback tier, retries) were exhausted, and the
    orchestrator has restored the safe baseline residency (planner +
    sketch) as a fallback. The caller's request failed, but the system
    is left in a known-good, user-visible-failure-free state."""

    def __init__(self, tier: str, attempts: int) -> None:
        self.tier = tier
        self.attempts = attempts
        super().__init__(
            f"OOM recovery exhausted for tier '{tier}' after {attempts} attempt(s). "
            "Baseline residency (planner + sketch) has been restored."
        )


class BackendLoadError(OrchestratorError):
    """A model backend's load() call failed for a reason other than OOM
    (missing weights, corrupt checkpoint, backend not implemented, etc.)."""


class StaleDesignStateError(OrchestratorError):
    """Raised on optimistic-concurrency conflict: the caller's expected
    revision doesn't match the stored revision."""

    def __init__(self, session_id: str, expected: int, actual: int) -> None:
        self.session_id = session_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Design state for session {session_id} is at revision {actual}, "
            f"caller expected {expected}. Re-read before writing."
        )


class SessionNotFoundError(OrchestratorError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"No session found with id {session_id}")
