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
  3. If retries are exhausted or budget preflight fails:
     recovery gives up on the *request* cleanly without silent degradation,
     while protecting the *system*: baseline residency (Planner + Sketch)
     is restored before control returns to the caller, and failure is reported.
     This makes behavior deterministic and testable: the specific call fails cleanly,
     and the orchestrator never gets stuck in a half-loaded state.
  4. Restoring baseline residency after ERROR_RECOVERY gets MORE retries
     than a normal load, not fewer — failing to restore Planner+Sketch is
     the one critical failure mode this design has, so it's worth
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
    high_vram: bool = False         # True -> use model_registry.get_registry
                                      # (high_vram=True): workstation 48GB profile
                                      # where all models are fully resident in VRAM.
    max_retries: int = 2
    load_timeout_s: float = 30.0
    baseline_restore_retries: int = 5   # extra attempts, per policy note (5) above
    backend_factory: Callable[[Any], ModelBackend] = default_backend_factory
    vram_safety_margin_gb: float = 0.0
    """Optional extra headroom subtracted from `envelope_gb` before every
    admission check (see vram_budget.py's _BaseLedger.safety_margin_gb).
    Defaults to 0.0 — no behavior change from prior versions. Operators running
    close to the edge on real hardware can raise this explicitly."""
    reproducible_experiment: bool = False

    state: StateMachine = field(init=False)
    registry: dict[Tier, Any] = field(init=False)
    ledger: VRAMLedger = field(init=False)
    ram_ledger: RAMLedger = field(init=False)
    backends: dict[Tier, ModelBackend] = field(init=False)
    _swap_lock: asyncio.Lock = field(init=False, repr=False)
    _started: bool = field(default=False, init=False)
    _warm_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _warm_status: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.state = StateMachine(ResidencyState.IDLE_RESIDENT)
        self.registry = get_registry(low_vram=self.low_vram, high_vram=self.high_vram)
        self.ledger = VRAMLedger(envelope_gb=self.envelope_gb, registry=self.registry, safety_margin_gb=self.vram_safety_margin_gb)
        self.ram_ledger = RAMLedger(envelope_gb=self.ram_envelope_gb, registry=self.registry)
        self.backends = {tier: self.backend_factory(spec) for tier, spec in self.registry.items()}
        self._swap_lock = asyncio.Lock()
        self._warm_task = None
        self._warm_status = {
            "state": "idle",
            "stage": "idle",
            "error": None,
            "loaded_tiers": [],
            "resident": [],
        }


    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def warm_status(self) -> dict[str, Any]:
        resident = [t.value for t in getattr(self.ledger, "resident_tiers", [])]
        self._warm_status["resident"] = resident
        return dict(self._warm_status)

    async def start_background_warm(self) -> asyncio.Task:
        """Start asynchronous non-blocking pre-warm of baseline tiers (Planner + Layout)."""
        if self._warm_task and not self._warm_task.done():
            return self._warm_task

        all_loaded = all(
            tier in getattr(self.ledger, "resident_tiers", set())
            for tier in ALWAYS_RESIDENT_TIERS
        )
        if all_loaded:
            self._warm_status = {
                "state": "ready",
                "stage": "ready",
                "error": None,
                "loaded_tiers": [t.value for t in ALWAYS_RESIDENT_TIERS],
                "resident": [t.value for t in getattr(self.ledger, "resident_tiers", [])],
            }
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(None)
            self._warm_task = asyncio.ensure_future(future)
            return self._warm_task

        self._warm_status = {
            "state": "warming",
            "stage": "starting",
            "error": None,
            "loaded_tiers": [],
            "resident": [t.value for t in getattr(self.ledger, "resident_tiers", [])],
        }
        self._warm_task = asyncio.create_task(self._do_background_warm())
        return self._warm_task

    async def _do_background_warm(self) -> None:
        try:
            for tier in ALWAYS_RESIDENT_TIERS:
                self._warm_status["stage"] = f"loading_{tier.value}"
                await self._load_one(tier)
                if tier.value not in self._warm_status["loaded_tiers"]:
                    self._warm_status["loaded_tiers"].append(tier.value)
            self._started = True
            self._warm_status["state"] = "ready"
            self._warm_status["stage"] = "ready"
            log.info("orchestrator_background_warm_complete", extra={"resident": self.ledger.snapshot()})
        except Exception as e:
            self._warm_status["state"] = "error"
            self._warm_status["stage"] = "failed"
            self._warm_status["error"] = str(e)
            log.error("orchestrator_background_warm_failed", extra={"error": str(e)})
            raise

    async def start(self) -> None:
        """Load baseline residency (Planner + Sketch). Call once at
        service startup."""
        for tier in ALWAYS_RESIDENT_TIERS:
            await self._load_one(tier)
            if tier.value not in self._warm_status["loaded_tiers"]:
                self._warm_status["loaded_tiers"].append(tier.value)
        self._started = True
        self._warm_status["state"] = "ready"
        self._warm_status["stage"] = "ready"
        log.info("orchestrator_started", extra={"resident": self.ledger.snapshot()})

    async def shutdown(self) -> None:
        if self._warm_task and not self._warm_task.done():
            self._warm_task.cancel()
            try:
                await self._warm_task
            except (asyncio.CancelledError, Exception):
                pass
        self._warm_task = None
        self._warm_status = {
            "state": "idle",
            "stage": "idle",
            "error": None,
            "loaded_tiers": [],
            "resident": [],
        }
        for tier in list(self.backends):
            if self.backends[tier].is_loaded:
                await self._unload_one(tier)
        self._started = False

    # ------------------------------------------------------------------ #
    # Conversational turn — §5.3 "Conversational turn"
    # ------------------------------------------------------------------ #

    def conversation_available(self) -> bool:
        return self.state.is_conversation_eligible() and not self._swap_lock.locked()

    @property
    def ready(self) -> bool:
        """Whether the baseline models have completed startup successfully."""
        return self._started and self.state.state == ResidencyState.IDLE_RESIDENT

    async def run_conversational_turn(self, **kwargs) -> Any:
        if self._warm_task and not self._warm_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._warm_task), timeout=self.load_timeout_s)
            except Exception as e:
                log.warning("conversational_turn_warm_wait_warning: %s", e)

        if not self.conversation_available():
            raise SwapBusyError(
                "Planner is not resident — a swap is in flight "
                f"(state={self.state.state.value}). Try again shortly."
            )
        planner_out = await self.backends[Tier.PLANNER].run(**kwargs)

        # BUG FOUND ON REVIEW (mid-turn constraint staleness): Sketch used
        # to receive the SAME `constraints` kwarg the caller passed in
        # BEFORE this turn's Planner call ran — meaning if the Planner's
        # OWN design_state_delta.constraint_updates changed style/
        # palette/layout_hints THIS turn (e.g. user says "make it dark
        # mode" and the Planner correctly extracts that), the Sketch
        # image generated in this SAME turn still used the OLD
        # constraints, since the caller (flows.py) only applies
        # constraint_updates onto persisted DesignState AFTER this whole
        # call returns — one turn too late for THIS turn's sketch. The
        # user would see a chat reply confirming "switching to dark
        # mode" next to a sketch that visibly isn't. Fixed by merging the
        # same delta the caller will apply afterward into a local copy of
        # `constraints` here too, via the shared apply_constraint_updates
        # helper (constraint_merge.py) — kept in a dependency-neutral
        # module specifically so this file still doesn't need to import
        # DesignState/pydantic to do this (see that module's docstring).
        from krisna_inference.orchestrator.constraint_merge import apply_constraint_updates

        delta = planner_out.get("design_state_delta") or {}
        constraint_updates = dict(delta.get("constraint_updates") or {})
        if "actions" in delta and "actions" not in constraint_updates:
            constraint_updates["actions"] = delta.get("actions")
        sketch_kwargs = dict(kwargs)
        if constraint_updates and isinstance(kwargs.get("constraints"), dict):
            sketch_kwargs["constraints"] = apply_constraint_updates(
                kwargs["constraints"], constraint_updates
            )

        sketch_out = await self.backends[Tier.SKETCH].run(**sketch_kwargs, planner_output=planner_out)
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
                if _is_oom_error(e):
                    log.error(
                        "tier_run_oom",
                        extra={"tier": active_tier.value if active_tier else "unknown", "error": str(e)},
                    )
                    if active_tier is not None:
                        await self._unload_one(active_tier)
                        active_tier = None
                    return await self._enter_recovery(preferred_tier, result.attempts + 1)
                raise
        except Exception as exc:
            log.error(
                "tier_run_failed",
                extra={"tier": active_tier.value if active_tier else "unknown", "error": str(exc)},
            )
            if active_tier is not None:
                await self._unload_one(active_tier)
                active_tier = None
            return await self._enter_recovery(preferred_tier, result.attempts + 1)
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
        """Run Planner Review in place of the retired Gemma Critic.

        Review is deliberately resident and shares the Planner contract.  The
        old Gemma path remains in the repository only for migration/reference
        purposes; it is no longer admitted by the product flow.
        """
        if not self.conversation_available():
            raise SwapBusyError("Planner is not resident — review is unavailable during a swap.")
        async with self._swap_lock:
            try:
                result = await self.backends[Tier.PLANNER].run(
                    message=run_kwargs.pop("message", "Review the current render and propose concrete fixes."),
                    constraints=run_kwargs.pop("constraints", {}),
                    conversation_history=run_kwargs.pop("conversation_history", None),
                    prior_critique=run_kwargs.pop("prior_critique", None),
                    review=True,
                    **run_kwargs,
                )
                return SwapResult(ok=True, tier_used=Tier.PLANNER, output=result)
            except Exception as e:
                return SwapResult(ok=False, tier_used=Tier.PLANNER, error=str(e))

    # ------------------------------------------------------------------ #
    # Load/unload primitives
    # ------------------------------------------------------------------ #

    async def _load_one(self, tier: Tier) -> None:
        if tier in getattr(self.ledger, "resident_tiers", set()) and self.backends[tier].is_loaded:
            return
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
        """Load requested tier cleanly with retry recovery, avoiding silent tier degradation."""
        total_attempts = 0
        if not self.ledger.would_fit(requested_tier) or not self.ram_ledger.would_fit(requested_tier):
            log.warning(
                "tier_budget_preflight_fail",
                extra={
                    "tier": requested_tier.value,
                    "vram_ledger": self.ledger.snapshot(),
                    "ram_ledger": self.ram_ledger.snapshot(),
                },
            )
            return await self._enter_recovery(requested_tier, attempts=0)

        for attempt in range(1, self.max_retries + 1):
            total_attempts += 1
            try:
                await self._load_one(requested_tier)
                return SwapResult(
                    ok=True, tier_used=requested_tier, degraded=False, attempts=total_attempts
                )
            except Exception as e:
                if _is_oom_error(e):
                    log.warning(
                        "tier_load_oom",
                        extra={"tier": requested_tier.value, "attempt": attempt, "error": str(e)},
                    )
                    # Real backend: torch.cuda.empty_cache() + gc.collect() here.
                    await asyncio.sleep(0.005 * attempt)
                    continue
                raise

        # All retries exhausted. Enter ERROR_RECOVERY,
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
