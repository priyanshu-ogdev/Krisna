"""SwapOrchestrator — drives ResidencyState transitions (§7.4) by actually
loading/unloading ModelBackend instances, enforcing the VRAM envelope, and
handling OOM per the policy documented below.

OOM handling policy (the part of §7.4 the PRD text was cut off before
reaching — see state_machine.py's module docstring for the same note):

  1. Before attempting a load, check VRAMLedger.would_fit(). If the
     *declared* budget alone doesn't fit, that's treated the same as a
     runtime OOM — no point attempting a load we already know won't fit.
  2. On a runtime OOM signal from the backend: clear/retry up to
     `max_retries` times (a real backend would call
     torch.cuda.empty_cache() + gc.collect() between attempts; MockBackend
     just counts attempts).
  3. If retries are exhausted and the tier has a `fallback_tier`
     (currently: POLISH_QUALITY -> POLISH_DEFAULT, per §6's stack table),
     attempt the fallback tier instead. The caller is told the requested
     tier was degraded, not just handed a silent substitution.
  4. If there's no fallback tier (CRITIC has none) or the fallback also
     fails, recovery gives up on the *request* but not the *system*:
     baseline residency (Planner + Sketch) is restored before control
     returns to the caller, and OOMRecoveryExhausted is raised. This is
     what makes "no user-visible failure" (§3) precise and testable: the
     specific finalize/critique call can fail, but the orchestrator itself
     never gets stuck in a half-loaded state that breaks the next
     conversational turn.
  5. Restoring baseline residency after ERROR_RECOVERY gets MORE retries
     than a normal load, not fewer — failing to restore Planner+Sketch is
     the one failure mode this design has no fallback for, so it's worth
     throwing extra attempts at before giving up.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from krisna_inference.orchestrator.exceptions import (
    BackendLoadError,
    OOMRecoveryExhausted,
    SwapBusyError,
    SwapTimeoutError,
)
from krisna_inference.orchestrator.model_registry import (
    ALWAYS_RESIDENT_TIERS,
    REGISTRY,
    ModelBackend,
    OOMSimulatedError,
    Tier,
    default_backend_factory,
    get_registry,
)
from krisna_inference.orchestrator.state_machine import ResidencyState, StateMachine, SwapTrigger
from krisna_inference.orchestrator.vram_budget import RAMLedger, VRAMLedger

log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).swap")


def _is_oom_error(exc: BaseException) -> bool:
    """Check whether an exception represents a simulated or runtime CUDA OOM."""
    if isinstance(exc, OOMSimulatedError):
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "oom" in msg)



@dataclass
class SwapResult:
    ok: bool
    tier_used: Tier | None
    degraded: bool = False           # true if a fallback tier was used instead of the requested one
    error: str | None = None
    attempts: int = 1
    output: Any = None


@dataclass
class SwapOrchestrator:
    envelope_gb: float = 24.0
    ram_envelope_gb: float = 48.0   # system-RAM budget for the CPU-offloaded
                                      # portion of any tier — only meaningful
                                      # when low_vram=True (every ram_gb is 0.0
                                      # otherwise, so this ledger is a no-op)
    low_vram: bool = False          # True -> use model_registry.get_registry
                                      # (low_vram=True): lower vram_gb figures,
                                      # populated ram_gb — see model_registry.py's
                                      # LOW_VRAM_REGISTRY for the real reasoning
                                      # behind each tier's numbers, and
                                      # inference/factory.py for how this flag
                                      # actually turns on CPU offload in each
                                      # real backend (registry numbers alone are
                                      # just bookkeeping — they don't make
                                      # offload happen on their own)
    max_retries: int = 2
    load_timeout_s: float = 30.0
    baseline_restore_retries: int = 5   # extra attempts, per policy note (5) above
    backend_factory: Callable[[Any], ModelBackend] = default_backend_factory
    vram_safety_margin_gb: float = 0.0
    """Optional extra headroom subtracted from `envelope_gb` before every
    admission check (see vram_budget.py's _BaseLedger.safety_margin_gb).
    Defaults to 0.0 — no behavior change from prior versions. Left as an
    explicit opt-in rather than a nonzero default because at least one
    registered tier (Critic, low-VRAM mode) is sized to fit its envelope
    with exactly zero headroom on paper; a nonzero default here would
    make that tier permanently inadmissible (it has no fallback_tier),
    which would trade a real problem for a worse one. Operators running
    close to the edge on real hardware should raise this explicitly, not
    rely on it being safe by default."""

    state: StateMachine = field(init=False)
    registry: dict[Tier, Any] = field(init=False)
    ledger: VRAMLedger = field(init=False)
    ram_ledger: RAMLedger = field(init=False)
    backends: dict[Tier, ModelBackend] = field(init=False)
    _swap_lock: asyncio.Lock = field(init=False, repr=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.state = StateMachine(ResidencyState.IDLE_RESIDENT)
        self.registry = get_registry(low_vram=self.low_vram)
        self.ledger = VRAMLedger(envelope_gb=self.envelope_gb, registry=self.registry, safety_margin_gb=self.vram_safety_margin_gb)
        self.ram_ledger = RAMLedger(envelope_gb=self.ram_envelope_gb, registry=self.registry)
        self.backends = {tier: self.backend_factory(spec) for tier, spec in self.registry.items()}
        self._swap_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Load baseline residency (Planner + Sketch). Call once at
        service startup."""
        for tier in ALWAYS_RESIDENT_TIERS:
            await self._load_one(tier)
        self._started = True
        log.info("orchestrator_started", extra={"resident": self.ledger.snapshot()})

    async def shutdown(self) -> None:
        for tier in list(self.backends):
            if self.backends[tier].is_loaded:
                await self._unload_one(tier)
        self._started = False

    # ------------------------------------------------------------------ #
    # Conversational turn — §5.3 "Conversational turn"
    # ------------------------------------------------------------------ #

    def conversation_available(self) -> bool:
        return self.state.is_conversation_eligible() and not self._swap_lock.locked()

    async def run_conversational_turn(self, **kwargs) -> Any:
        if not self.conversation_available():
            raise SwapBusyError(
                "Planner is not resident — a swap is in flight "
                f"(state={self.state.state.value}). Try again shortly."
            )
        planner_out = await self.backends[Tier.PLANNER].run(**kwargs)
        sketch_out = await self.backends[Tier.SKETCH].run(**kwargs, planner_output=planner_out)
        return {"planner": planner_out, "sketch": sketch_out}

    # ------------------------------------------------------------------ #
    # Finalize — §5.3 "Finalize"
    # ------------------------------------------------------------------ #

    async def request_finalize(
        self, preferred_tier: Tier = Tier.POLISH_DEFAULT, **run_kwargs
    ) -> SwapResult:
        if self._swap_lock.locked():
            raise SwapBusyError("Another swap is already in flight.")
        async with self._swap_lock:
            return await self._finalize_locked(preferred_tier, **run_kwargs)

    async def _finalize_locked(self, preferred_tier: Tier, **run_kwargs) -> SwapResult:
        self.state.apply(SwapTrigger.FINALIZE_REQUESTED)

        # Unload baseline before loading the polish tier — this is the
        # step that makes the 16-24GB envelope achievable at all (§5.3:
        # "Orchestrator unloads Planner + Sketch Tier" happens BEFORE
        # "Orchestrator loads chosen Polish Tier").
        await self._unload_baseline()

        result = await self._load_with_recovery(preferred_tier)
        if not result.ok:
            return result  # baseline already restored inside _load_with_recovery on failure

        self.state.apply(SwapTrigger.POLISH_LOAD_COMPLETE)

        active_tier: Tier | None = result.tier_used
        output = None
        try:
            try:
                output = await self.backends[active_tier].run(**run_kwargs)
            except Exception as e:
                fallback = self.registry[active_tier].fallback_tier if active_tier else None
                if _is_oom_error(e) and fallback is not None:
                    log.warning(
                        "tier_run_oom_fallback",
                        extra={
                            "tier": active_tier.value,
                            "fallback": fallback.value,
                            "error": str(e),
                        },
                    )
                    # Unload failed tier and attempt fallback
                    await self._unload_one(active_tier)
                    active_tier = None

                    # Load fallback tier
                    await self._load_one(fallback)
                    active_tier = fallback
                    output = await self.backends[fallback].run(**run_kwargs)
                    result.tier_used = fallback
                    result.degraded = True
                elif _is_oom_error(e):
                    log.warning(
                        "tier_run_oom_no_fallback",
                        extra={"tier": active_tier.value if active_tier else "unknown", "error": str(e)},
                    )
                    if active_tier is not None:
                        await self._unload_one(active_tier)
                        active_tier = None
                    return await self._enter_recovery(preferred_tier, result.attempts + 1)
                else:
                    raise
        except Exception as exc:
            if _is_oom_error(exc):
                log.warning(
                    "tier_fallback_run_oom",
                    extra={"error": str(exc)},
                )
                if active_tier is not None:
                    await self._unload_one(active_tier)
                    active_tier = None
                return await self._enter_recovery(preferred_tier, result.attempts + 2)
            raise
        finally:
            if self.state.state == ResidencyState.POLISH_RESIDENT:
                self.state.apply(SwapTrigger.POLISH_DONE)
                if active_tier is not None:
                    await self._unload_one(active_tier)
                await self._restore_baseline()
                self.state.apply(SwapTrigger.BASELINE_RESTORED)

        result.output = output
        return result

    # ------------------------------------------------------------------ #
    # Critique — §5.3 "Critique pass"
    # ------------------------------------------------------------------ #

    async def request_critique(self, **run_kwargs) -> SwapResult:
        if self._swap_lock.locked():
            raise SwapBusyError("Another swap is already in flight.")
        async with self._swap_lock:
            return await self._critique_locked(**run_kwargs)

    async def _critique_locked(self, **run_kwargs) -> SwapResult:
        self.state.apply(SwapTrigger.CRITIQUE_REQUESTED)
        await self._unload_baseline()

        result = await self._load_with_recovery(Tier.CRITIC)
        if not result.ok:
            return result

        self.state.apply(SwapTrigger.CRITIC_LOAD_COMPLETE)
        active_tier: Tier | None = Tier.CRITIC
        output = None
        try:
            try:
                output = await self.backends[Tier.CRITIC].run(**run_kwargs)
            except Exception as e:
                if _is_oom_error(e):
                    log.warning(
                        "critic_run_oom",
                        extra={"tier": Tier.CRITIC.value, "error": str(e)},
                    )
                    if active_tier is not None:
                        await self._unload_one(active_tier)
                        active_tier = None
                    return await self._enter_recovery(Tier.CRITIC, result.attempts + 1)
                raise
        finally:
            if self.state.state == ResidencyState.CRITIC_RESIDENT:
                self.state.apply(SwapTrigger.CRITIC_DONE)
                if active_tier is not None:
                    await self._unload_one(active_tier)
                await self._restore_baseline()
                self.state.apply(SwapTrigger.BASELINE_RESTORED)

        result.output = output
        return result

    # ------------------------------------------------------------------ #
    # Load/unload primitives
    # ------------------------------------------------------------------ #

    async def _load_one(self, tier: Tier) -> None:
        spec = self.registry[tier]
        try:
            self.ledger.admit(tier)
            try:
                self.ram_ledger.admit(tier)
            except Exception:
                # VRAM admitted but RAM budget blown — roll back the VRAM
                # reservation too, so a RAM-budget failure doesn't leave a
                # phantom VRAM reservation with no backing load. This is
                # the two-ledger equivalent of the single-ledger rollback
                # in the except blocks below.
                self.ledger.release(tier)
                raise
        except Exception:
            # would_fit()/admit() budget failure is handled by callers that
            # care (_load_with_recovery); a bare _load_one (e.g. startup)
            # should surface it directly.
            raise
        try:
            await asyncio.wait_for(self.backends[tier].load(), timeout=self.load_timeout_s)
        except asyncio.TimeoutError as e:
            self.ledger.release(tier)
            self.ram_ledger.release(tier)
            raise SwapTimeoutError(tier.value, "load", self.load_timeout_s) from e
        except Exception:
            self.ledger.release(tier)
            self.ram_ledger.release(tier)
            raise
        log.info(
            "tier_loaded",
            extra={"tier": tier.value, "vram_gb": spec.vram_gb, "ram_gb": spec.ram_gb},
        )

    async def _unload_one(self, tier: Tier) -> None:
        if not self.backends[tier].is_loaded:
            self.ledger.release(tier)
            self.ram_ledger.release(tier)
            return
        try:
            await asyncio.wait_for(self.backends[tier].unload(), timeout=self.load_timeout_s)
        except asyncio.TimeoutError as e:
            # Unload timing out is worse than load timing out — we don't
            # know if VRAM was actually freed. Log loudly; still release
            # our own ledger accounting so budget math doesn't wedge, but
            # this is exactly the kind of thing real monitoring should
            # alert on.
            log.error("tier_unload_timeout", extra={"tier": tier.value})
            raise SwapTimeoutError(tier.value, "unload", self.load_timeout_s) from e
        finally:
            self.ledger.release(tier)
            self.ram_ledger.release(tier)
        log.info("tier_unloaded", extra={"tier": tier.value})

    async def _unload_baseline(self) -> None:
        for tier in ALWAYS_RESIDENT_TIERS:
            await self._unload_one(tier)

    async def _restore_baseline(self, retries: int | None = None) -> None:
        """Reload Planner + Sketch. This is the one operation this design
        has no fallback for, so it gets extra retries (policy note 5)."""
        attempts = retries if retries is not None else self.baseline_restore_retries
        for tier in ALWAYS_RESIDENT_TIERS:
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    await self._load_one(tier)
                    last_exc = None
                    break
                except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
                    last_exc = e
                    log.warning(
                        "baseline_restore_retry",
                        extra={"tier": tier.value, "attempt": attempt, "error": str(e)},
                    )
                    await asyncio.sleep(0.01 * attempt)
            if last_exc is not None:
                # Every retry exhausted for a baseline tier. This is the
                # worst-case failure this system can hit — surface it
                # loudly rather than pretending the system is healthy.
                log.critical(
                    "baseline_restore_failed",
                    extra={"tier": tier.value, "attempts": attempts},
                )
                raise BackendLoadError(
                    f"Could not restore baseline tier '{tier.value}' after "
                    f"{attempts} attempts. System is NOT in a safe state."
                ) from last_exc

    async def _load_with_recovery(self, requested_tier: Tier) -> SwapResult:
        """Implements the OOM policy documented in this module's docstring."""
        candidates = [requested_tier]
        fallback = self.registry[requested_tier].fallback_tier
        if fallback is not None:
            candidates.append(fallback)

        total_attempts = 0
        for candidate in candidates:
            degraded = candidate != requested_tier
            if not self.ledger.would_fit(candidate) or not self.ram_ledger.would_fit(candidate):
                log.warning(
                    "tier_budget_preflight_fail",
                    extra={
                        "tier": candidate.value,
                        "vram_ledger": self.ledger.snapshot(),
                        "ram_ledger": self.ram_ledger.snapshot(),
                    },
                )
                continue  # try next candidate (fallback), same as an OOM

            for attempt in range(1, self.max_retries + 1):
                total_attempts += 1
                try:
                    await self._load_one(candidate)
                    return SwapResult(
                        ok=True, tier_used=candidate, degraded=degraded, attempts=total_attempts
                    )
                except OOMSimulatedError as e:
                    log.warning(
                        "tier_load_oom",
                        extra={"tier": candidate.value, "attempt": attempt, "error": str(e)},
                    )
                    # Real backend: torch.cuda.empty_cache() + gc.collect() here.
                    await asyncio.sleep(0.005 * attempt)
                    continue

        # All candidates and all retries exhausted. Enter ERROR_RECOVERY,
        # restore baseline, and report failure — but the system stays healthy.
        return await self._enter_recovery(requested_tier, total_attempts)

    async def _enter_recovery(self, requested_tier: Tier, attempts: int) -> SwapResult:
        self.state.apply(SwapTrigger.OOM)
        try:
            await self._restore_baseline()
        finally:
            self.state.apply(SwapTrigger.RECOVERY_COMPLETE)
        exc = OOMRecoveryExhausted(requested_tier.value, attempts)
        log.error(
            "oom_recovery_exhausted",
            extra={"tier": requested_tier.value, "attempts": attempts},
        )
        return SwapResult(ok=False, tier_used=None, error=str(exc), attempts=attempts)
