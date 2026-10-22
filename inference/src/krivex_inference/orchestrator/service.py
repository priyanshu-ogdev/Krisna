"""Packaged local service for the Swap Orchestrator + Design State Manager.

Run with:  uvicorn krivex_inference (formerly krisna_orchestrator).service:app --host 127.0.0.1 --port 8420
(or use scripts/run_service.sh)

This wraps swap_orchestrator.SwapOrchestrator + store.DesignStateStore +
flows.py behind a small HTTP API so the rest of the system (a future UI,
the actual model-loading layer, or a test harness) can drive sessions
without importing Python internals directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from krivex_inference.orchestrator import flows
from krivex_inference.orchestrator.design_state import DesignState
from krivex_training.dpo.preference_store import PreferenceStore
from krivex_inference.orchestrator.exceptions import (
    OOMRecoveryExhausted,
    OrchestratorError,
    SessionNotFoundError,
    StaleDesignStateError,
    SwapBusyError,
)
from krivex_inference.orchestrator.flows import FlowError
from krivex_inference.orchestrator.logging_setup import configure_logging
from krivex_inference.orchestrator.store import DesignStateStore
from krivex_inference.orchestrator.swap_orchestrator import SwapOrchestrator
from krivex_inference.orchestrator.session_coordinator import SessionCoordinator

configure_logging()
log = logging.getLogger("krivex_inference (formerly krisna_orchestrator).service")

if os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1":
    from krivex_inference.backends.factory import real_backend_factory

    _low_vram = os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"
    _high_vram = os.environ.get("KRISNA_HIGH_VRAM_MODE") == "1"

    _default_vram = "12.0" if _low_vram else ("48.0" if _high_vram else "24.0")
    _default_ram = "128.0" if _high_vram else "64.0"

    orchestrator = SwapOrchestrator(
        backend_factory=real_backend_factory,
        low_vram=_low_vram,
        high_vram=_high_vram,
        envelope_gb=float(os.environ.get("KRISNA_VRAM_ENVELOPE_GB", _default_vram)),
        ram_envelope_gb=float(os.environ.get("KRISNA_RAM_ENVELOPE_GB", _default_ram)),
    )
    log.warning(
        "using_real_inference_backends",
        extra={
            "note": "GPU + downloaded weights required — see README's inference-layer section",
            "low_vram_mode": _low_vram,
            "high_vram_mode": _high_vram,
            "vram_envelope_gb": orchestrator.envelope_gb,
            "ram_envelope_gb": orchestrator.ram_envelope_gb if _low_vram else None,
        },
    )
    from krivex_inference.verifiers.verifier_stack import get_verifier_stack

    _verifier_stack = get_verifier_stack()

    # UPGRADE: real, always-crashing bug found during an I/O-contract
    # audit of the Finalize flow. sketch_handoff.py's
    # make_vq_decode_handoff() — the actual "VQ tokens -> decoded pixel
    # image" conversion Finalize requires before Polish ever runs — was
    # fully implemented and correct, but never imported or called
    # anywhere outside its own tests. flows.finalize()'s `handoff_hook`
    # therefore defaulted to identity passthrough on every real call:
    # state.sketch_tokens.vq_tokens (a "blob://tokens_xxx.json" ref, per
    # BlobStore.save_tokens — a JSON file of token IDs) was passed
    # straight through as `handoff_image_ref` to the Polish backend,
    # which calls `store.load_image()` on it -> `PIL.Image.open()` on a
    # JSON text file -> guaranteed UnidentifiedImageError on every single
    # real /finalize call. Not a hypothetical edge case — this was the
    # actual, only code path service.py used.
    from krivex_inference.backends.blob_store_singleton import get_blob_store
    # Structured Sketch emits a deterministic wireframe/image reference
    # directly. There is no VQ token decode stage in the consumer release.
    def _handoff_hook(layout_ref: str) -> str:
        return layout_ref
else:
    orchestrator = SwapOrchestrator()  # MockBackend — safe default, no GPU/weights needed
    _verifier_stack = None
    _handoff_hook = None  # flows.finalize()'s identity-passthrough default is fine for MockBackend — its polish backends don't call PIL.Image.open() on the ref at all.

store = DesignStateStore(db_path="krisna_sessions.db")
preference_store = PreferenceStore(db_path="krisna_preference_pairs.db")
session_coordinator = SessionCoordinator(
    inactivity_seconds=float(os.environ.get("KRISNA_SELF_REVIEW_IDLE_SECONDS", "30"))
)


_session_locks: dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


async def _run_scheduled_self_review(session_id: str) -> None:
    async with _get_session_lock(session_id):
        try:
            state = store.get(session_id)
        except Exception:
            return
        if state.refinement.exhausted or not state.layout.layout_ref:
            return
        try:
            await flows.planner_self_review(orchestrator, store, session_id)
            updated = store.get(session_id)
            if not updated.refinement.exhausted and updated.stage.value in ("sketching", "conversing"):
                _schedule_self_review(session_id)
        except Exception as e:
            log.warning("scheduled_self_review_skipped_or_failed", extra={"session_id": session_id, "error": str(e)})


def _schedule_self_review(session_id: str) -> None:
    state = store.get(session_id)
    if state.refinement.exhausted or not state.layout.layout_ref:
        return
    state.refinement.self_review_scheduled = True
    state.record_refinement_event("self_review_scheduled", "system")
    state.touch()
    store.save(state, state.revision - 1)
    session_coordinator.user_activity(session_id, lambda: _run_scheduled_self_review(session_id))


@asynccontextmanager
async def lifespan(_: FastAPI):
    await orchestrator.start()
    log.info("service_ready")
    yield
    await session_coordinator.stop()
    await orchestrator.shutdown()
    preference_store.close()


app = FastAPI(title="Krisna Swap Orchestrator", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------- #

class NewSessionRequest(BaseModel):
    style: str | None = None
    palette: list[str] = []


class MessageRequest(BaseModel):
    message: str
    image_ref: str | None = None
    auto_proceed: bool = False
    quality: bool = False
    fast: bool = False
    prompt: str | None = None
    auto_review: bool = True


class ExecuteRequest(BaseModel):
    message: str
    image_ref: str | None = None
    quality: bool = False
    fast: bool = False
    prompt: str | None = None
    auto_review: bool = True


class DecompileRequest(BaseModel):
    image_ref: str
    instruction: str = "Decompile this UI screenshot into an editable layout."


class FinalizeRequest(BaseModel):
    quality: bool = False
    prompt: str | None = None


class EnhanceRequest(BaseModel):
    image_ref: str
    prompt: str
    quality: bool = True


class RefineRequest(BaseModel):
    instruction: str
    quality: bool = False
    num_inference_steps: int = 8
    actions: list[Any] = Field(default_factory=list)


class PreviewRequest(BaseModel):
    prompt: str | None = None


class CritiqueRequest(BaseModel):
    compare_against_image_ref: str | None = None
    compare_against_score: float | None = None


class WarmRequest(BaseModel):
    background: bool = False


def _error_response(e: Exception) -> HTTPException:
    if isinstance(e, SessionNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, SwapBusyError):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, (StaleDesignStateError, FlowError)):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, OOMRecoveryExhausted):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, OrchestratorError):
        return HTTPException(status_code=500, detail=str(e))
    log.exception("unhandled_error")
    return HTTPException(status_code=500, detail="internal error")


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": orchestrator.ready,
        "backend_mode": "real" if os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1" else "mock",
        "residency_state": orchestrator.state.state.value,
        "vram": orchestrator.ledger.snapshot(),
    }


@app.get("/ready")
async def ready():
    """Readiness probe: checks if orchestrator is running and state store is accessible."""
    is_ready = orchestrator.ready
    return {
        "ready": is_ready,
        "backend_mode": "real" if os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1" else "mock",
        "residency_state": orchestrator.state.state.value,
        "store_accessible": True,
    }


@app.get("/models/status")
async def models_status():
    """Detailed model artifact discovery, memory allocations, and hardware status."""
    from krivex_inference.backends.model_loader import (
        discover_planner_model,
        discover_mmproj_model,
        probe_cuda_hardware,
    )
    from krivex_inference.orchestrator.vram_budget import probe_real_ram, probe_real_vram

    planner_path = discover_planner_model()
    mmproj_path = discover_mmproj_model()
    hw = probe_cuda_hardware()

    return {
        "residency_state": orchestrator.state.state.value,
        "resident_tiers": [t.value for t in getattr(orchestrator.ledger, "resident_tiers", [])],
        "planner": {
            "discovered_path": str(planner_path) if planner_path else None,
            "mmproj_path": str(mmproj_path) if mmproj_path else None,
            "runtime": "llama.cpp",
        },
        "polish": {
            "model_id": os.environ.get("KRISNA_POLISH_MODEL_ID", "Qwen/Qwen-Image-2.1"),
            "lora_adapter": os.environ.get("KRISNA_POLISH_LORA_PATH") or os.environ.get("KRISNA_POLISH_DEFAULT_LORA_PATH"),
            "fast_lora_adapter": os.environ.get("KRISNA_POLISH_FAST_LORA_PATH"),
        },
        "vram_ledger": orchestrator.ledger.snapshot(),
        "ram_ledger": orchestrator.ram_ledger.snapshot(),
        "real_vram": probe_real_vram(),
        "real_ram": probe_real_ram(),
        "cuda_hardware": hw,
        "low_vram_mode": orchestrator.low_vram,
    }


@app.post("/models/warm")
async def warm_models(req: WarmRequest | None = None):
    """Pre-warm baseline tiers (Planner + Layout) synchronously or in background."""
    req = req or WarmRequest()
    try:
        if req.background:
            await orchestrator.start_background_warm()
            return {
                "status": "warming",
                "background": True,
                "warm_status": orchestrator.warm_status,
            }
        await orchestrator.start()
        return {
            "status": "warmed",
            "background": False,
            "residency_state": orchestrator.state.state.value,
            "resident_tiers": [t.value for t in getattr(orchestrator.ledger, "resident_tiers", [])],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@app.get("/models/warm/status")
async def warm_models_status():
    """Poll asynchronous pre-warming progress for baseline tiers."""
    return orchestrator.warm_status


@app.post("/models/unload")
async def unload_models():
    """Unload all resident models and release GPU cache for external resource reclamation."""
    try:
        await orchestrator.shutdown()
        return {
            "status": "unloaded",
            "residency_state": orchestrator.state.state.value,
            "resident_tiers": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unload failed: {e}")


@app.post("/session")
async def create_session(req: NewSessionRequest) -> dict:
    state = DesignState()
    state.constraints.style = req.style
    state.constraints.palette = req.palette
    saved = store.create(state)
    return saved.to_wire() | {"revision": saved.revision}


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as e:
        raise _error_response(e)
    return state.to_wire() | {"revision": state.revision}


@app.get("/session/{session_id}/render")
async def get_session_render(session_id: str):
    """Serves the session's finalized image as actual bytes.

    Added alongside the frontend's pixel-forming visualization —
    finalize_output.image_ref (a `blob://...` string) was never
    fetchable by a browser before this: DesignState.to_wire() only ever
    returned the *reference*, and this service had no route that
    resolved it to real bytes. Without this, no UI could ever show the
    actual rendered image, only its existence. Session-scoped (routed
    through the same session_id every other endpoint uses) rather than
    a generic /blob/{ref} route, so a client can only ever fetch the
    image belonging to the session it already has access to — image_ref
    itself is never accepted as a request parameter, only read
    server-side from that session's own state, which also rules out any
    path-traversal concern (BlobStore.path_for()'s ref always comes from
    save_image()'s own uuid-based filename, never client input).
    """
    from fastapi.responses import FileResponse

    from krivex_inference.backends.blob_store_singleton import get_blob_store

    try:
        state = store.get(session_id)
    except SessionNotFoundError as e:
        raise _error_response(e)
    image_ref = state.finalize_output.image_ref
    if not image_ref or not image_ref.startswith("blob://"):
        raise HTTPException(status_code=404, detail="No finalized image for this session yet.")
    path = get_blob_store().path_for(image_ref)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image reference exists but the blob is missing on disk.")
    return FileResponse(path, media_type="image/png")


@app.get("/session/{session_id}/model-status")
async def get_session_model_status(session_id: str) -> dict:
    """Query model residency readiness and asynchronous warm status for this session."""
    try:
        _ = store.get(session_id)
    except SessionNotFoundError as e:
        raise _error_response(e)
    return {
        "session_id": session_id,
        "ready": orchestrator.conversation_available(),
        "residency_state": orchestrator.state.state.value,
        "warm_status": orchestrator.warm_status,
    }


@app.post("/session/{session_id}/message")
async def post_message(session_id: str, req: MessageRequest) -> dict:
    session_coordinator.cancel(session_id)
    async with _get_session_lock(session_id):
        try:
            if req.auto_proceed:
                state = await flows.plan_sketch_and_polish(
                    orchestrator,
                    store,
                    session_id,
                    user_message=req.message,
                    image_ref=req.image_ref,
                    quality=req.quality,
                    fast=req.fast,
                    prompt=req.prompt,
                    verifier_stack=_verifier_stack,
                    handoff_hook=_handoff_hook,
                    auto_review=req.auto_review,
                )
            else:
                state = await flows.conversational_turn(
                    orchestrator, store, session_id, req.message, image_ref=req.image_ref
                )
                _schedule_self_review(session_id)
        except (SessionNotFoundError, SwapBusyError, StaleDesignStateError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        res = state.to_wire() | {"revision": state.revision}
        if state.preview_output:
            res["preview_output"] = state.preview_output.model_dump()
        return res


@app.post("/session/{session_id}/execute")
async def post_execute(session_id: str, req: ExecuteRequest) -> dict:
    """Unified simultaneous Plan + Sketch, direct-to-Polish, Verifier Review, and Further Planning."""
    session_coordinator.cancel(session_id)
    async with _get_session_lock(session_id):
        try:
            state = await flows.plan_sketch_and_polish(
                orchestrator,
                store,
                session_id,
                user_message=req.message,
                image_ref=req.image_ref,
                quality=req.quality,
                fast=req.fast,
                prompt=req.prompt,
                verifier_stack=_verifier_stack,
                handoff_hook=_handoff_hook,
                auto_review=req.auto_review,
            )
        except (SessionNotFoundError, SwapBusyError, StaleDesignStateError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        res = state.to_wire() | {"revision": state.revision}
        if state.preview_output:
            res["preview_output"] = state.preview_output.model_dump()
        return res


@app.post("/session/{session_id}/decompile")
async def post_decompile(session_id: str, req: DecompileRequest) -> dict:
    """VLM Screenshot-to-Layout Decompilation (ACL Design2Code / CVPR ShowUI)."""
    session_coordinator.cancel(session_id)
    async with _get_session_lock(session_id):
        try:
            state = await flows.decompile_screenshot(
                orchestrator, store, session_id, image_ref=req.image_ref, instruction=req.instruction
            )
        except (SessionNotFoundError, SwapBusyError, StaleDesignStateError, FlowError) as e:
            raise _error_response(e)
        _schedule_self_review(session_id)
        return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/self-review")
async def post_self_review(session_id: str) -> dict:
    """Run one autonomous Planner/Layout refinement turn.

    A future session coordinator will call this after inactivity; exposing the
    same operation now keeps the policy and HTTP behavior testable.
    """
    session_coordinator.cancel(session_id)
    async with _get_session_lock(session_id):
        try:
            state = await flows.planner_self_review(orchestrator, store, session_id)
        except (SessionNotFoundError, SwapBusyError, StaleDesignStateError, FlowError) as e:
            raise _error_response(e)
        return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/finalize")
async def post_finalize(session_id: str, req: FinalizeRequest) -> dict:
    async with _get_session_lock(session_id):
        try:
            state = await flows.finalize(
                orchestrator, store, session_id,
                quality=req.quality, prompt=req.prompt, verifier_stack=_verifier_stack,
                handoff_hook=_handoff_hook,
            )
        except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/enhance")
async def post_enhance(session_id: str, req: EnhanceRequest) -> dict:
    async with _get_session_lock(session_id):
        try:
            state = await flows.enhance_photo(
                orchestrator, store, session_id,
                image_ref=req.image_ref, prompt=req.prompt, quality=req.quality
            )
        except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/refine")
async def post_refine(session_id: str, req: RefineRequest) -> dict:
    """FlowEdit-Style Iterative Refinement (Kulikov et al., ICCV 2025)."""
    async with _get_session_lock(session_id):
        try:
            state = await flows.iterative_refine(
                orchestrator,
                store,
                session_id,
                instruction=req.instruction,
                quality=req.quality,
                num_inference_steps=req.num_inference_steps,
                verifier_stack=_verifier_stack,
                actions=req.actions,
            )
        except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/preview")
async def post_preview(session_id: str, req: PreviewRequest | None = None) -> dict:
    """Instant 4-8 step preview using Consistency Flow Matching (Tier.POLISH_FAST)."""
    req = req or PreviewRequest()
    async with _get_session_lock(session_id):
        try:
            state = await flows.fast_preview(
                orchestrator,
                store,
                session_id,
                prompt=req.prompt,
                verifier_stack=_verifier_stack,
                handoff_hook=_handoff_hook,
            )
        except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        res = state.to_wire() | {"revision": state.revision}
        if state.preview_output:
            res["preview_output"] = state.preview_output.model_dump()
        return res


@app.post("/session/{session_id}/critique")
async def post_critique(session_id: str, req: CritiqueRequest | None = None) -> dict:
    req = req or CritiqueRequest()
    compare_against = None
    if req.compare_against_image_ref is not None and req.compare_against_score is not None:
        compare_against = {"image_ref": req.compare_against_image_ref, "score": req.compare_against_score}
    async with _get_session_lock(session_id):
        try:
            state = await flows.critique_pass(
                orchestrator, store, session_id,
                preference_store=preference_store, compare_against=compare_against,
            )
        except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
            raise _error_response(e)
        return state.to_wire() | {"revision": state.revision}


@app.get("/preference-pairs/stats")
async def preference_pairs_stats() -> dict:
    planner_review_count = preference_store.count(source="planner_review")
    return {
        "total": preference_store.count(),
        "verifier_stack": preference_store.count(source="verifier_stack"),
        # Deprecated compatibility key: older UI clients called this bucket
        # gemma_critique. New writes are explicitly planner_review and never
        # use frozen-Critic self-labels, but the response shape stays stable.
        "gemma_critique": planner_review_count,
        "uicrit_seed": preference_store.count(source="uicrit_seed"),
    }


@app.post("/preference-pairs/export")
async def preference_pairs_export(source: str | None = None) -> dict:
    from krivex_training.dpo.dpo_dataset_export import export_jsonl

    output_path = f"krisna_dpo_export{'_' + source if source else ''}.jsonl"
    count = export_jsonl(preference_store, output_path, source=source)
    return {"written": count, "path": output_path}


@app.get("/orchestrator/status")
async def orchestrator_status() -> dict:
    from krivex_inference.orchestrator.vram_budget import probe_real_ram, probe_real_vram

    return {
        "residency_state": orchestrator.state.state.value,
        "conversation_available": orchestrator.conversation_available(),
        "ready": orchestrator.ready,
        "vram": orchestrator.ledger.snapshot(),
        # New: system-RAM ledger — always present (never None), since
        # it's a well-defined no-op snapshot in full-VRAM mode (every
        # resident spec has ram_gb=0.0) rather than something that only
        # exists conditionally on low_vram — a caller checking this
        # field doesn't need to know which mode is active first.
        "ram": orchestrator.ram_ledger.snapshot(),
        # New: real, live hardware readings (torch.cuda.mem_get_info(),
        # /proc/meminfo) alongside the declared ledger numbers above.
        # probe_real_vram()/probe_real_ram() already existed in
        # vram_budget.py for diagnostics/logging — confirmed by grep
        # they were never actually wired into any API response before
        # this, so no client (this project's own frontend included)
        # could ever show real hardware utilization, only the admission
        # ledger's own declared bookkeeping. None on hosts without
        # CUDA/without /proc (e.g. this endpoint running the MockBackend
        # dev path on a Mac) — a null here means "not available on this
        # host," never "zero used."
        "real_vram": probe_real_vram(),
        "real_ram": probe_real_ram(),
        "low_vram_mode": orchestrator.low_vram,
        "history_len": len(orchestrator.state.history),
    }


@app.get("/session/{session_id}/research-trace")
async def get_research_trace(session_id: str) -> dict:
    """Exports structured reasoning traces, VLA action history, and verifier evaluations
    for academic paper analysis and ablation studies.
    """
    try:
        trajectory = flows.export_research_trajectory(store, session_id)
        return trajectory.model_dump()
    except SessionNotFoundError as e:
        raise _error_response(e)


def main() -> None:
    """CLI entry point for running the Krisna Inference FastAPI service."""
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="krivex-inference",
        description="Krisna Inference Swap Orchestrator FastAPI service",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("KRISNA_BIND_HOST", "127.0.0.1"),
        help="Host to bind service to (default: 127.0.0.1 or KRISNA_BIND_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KRISNA_PORT", "8420")),
        help="Port to bind service to (default: 8420 or KRISNA_PORT)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for development",
    )
    args = parser.parse_args()

    uvicorn.run(
        "krivex_inference.orchestrator.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
