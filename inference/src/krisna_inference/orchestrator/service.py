"""Packaged local service for the Swap Orchestrator + Design State Manager.

Run with:  uvicorn krisna_inference (formerly krisna_orchestrator).service:app --host 127.0.0.1 --port 8420
(or use scripts/run_service.sh)

This wraps swap_orchestrator.SwapOrchestrator + store.DesignStateStore +
flows.py behind a small HTTP API so the rest of the system (a future UI,
the actual model-loading layer, or a test harness) can drive sessions
without importing Python internals directly.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from krisna_inference.orchestrator import flows
from krisna_inference.orchestrator.design_state import DesignState
from krisna_training.dpo.preference_store import PreferenceStore
from krisna_inference.orchestrator.exceptions import (
    OOMRecoveryExhausted,
    OrchestratorError,
    SessionNotFoundError,
    StaleDesignStateError,
    SwapBusyError,
)
from krisna_inference.orchestrator.flows import FlowError
from krisna_inference.orchestrator.logging_setup import configure_logging
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator

configure_logging()
log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).service")

if os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1":
    from krisna_inference.backends.factory import real_backend_factory

    _low_vram = os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"
    orchestrator = SwapOrchestrator(
        backend_factory=real_backend_factory,
        low_vram=_low_vram,
        # 12.0 is the low-VRAM target this README documents; override via
        # KRISNA_VRAM_ENVELOPE_GB if your card/target differs. Only takes
        # effect together with low_vram=True — the full-VRAM registry's
        # vram_gb values (up to 18GB for Critic alone) won't fit a 12GB
        # envelope at all, so envelope_gb is read regardless of low_vram
        # but only makes practical sense lowered when low_vram is also on.
        envelope_gb=float(os.environ.get("KRISNA_VRAM_ENVELOPE_GB", "12.0" if _low_vram else "24.0")),
        ram_envelope_gb=float(os.environ.get("KRISNA_RAM_ENVELOPE_GB", "64.0")),
    )
    log.warning(
        "using_real_inference_backends",
        extra={
            "note": "GPU + downloaded weights required — see README's inference-layer section",
            "low_vram_mode": _low_vram,
            "vram_envelope_gb": orchestrator.envelope_gb,
            "ram_envelope_gb": orchestrator.ram_envelope_gb if _low_vram else None,
        },
    )
    from krisna_inference.verifiers.verifier_stack import get_verifier_stack

    _verifier_stack = get_verifier_stack()
else:
    orchestrator = SwapOrchestrator()  # MockBackend — safe default, no GPU/weights needed
    _verifier_stack = None

store = DesignStateStore(db_path="krisna_sessions.db")
preference_store = PreferenceStore(db_path="krisna_preference_pairs.db")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await orchestrator.start()
    log.info("service_ready")
    yield
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


class FinalizeRequest(BaseModel):
    quality: bool = False
    prompt: str | None = None


class CritiqueRequest(BaseModel):
    compare_against_image_ref: str | None = None
    compare_against_score: float | None = None


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
        "residency_state": orchestrator.state.state.value,
        "vram": orchestrator.ledger.snapshot(),
    }


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


@app.post("/session/{session_id}/message")
async def post_message(session_id: str, req: MessageRequest) -> dict:
    try:
        state = await flows.conversational_turn(orchestrator, store, session_id, req.message)
    except (SessionNotFoundError, SwapBusyError, StaleDesignStateError) as e:
        raise _error_response(e)
    return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/finalize")
async def post_finalize(session_id: str, req: FinalizeRequest) -> dict:
    try:
        state = await flows.finalize(
            orchestrator, store, session_id,
            quality=req.quality, prompt=req.prompt, verifier_stack=_verifier_stack,
        )
    except (SessionNotFoundError, SwapBusyError, FlowError, OOMRecoveryExhausted) as e:
        raise _error_response(e)
    return state.to_wire() | {"revision": state.revision}


@app.post("/session/{session_id}/critique")
async def post_critique(session_id: str, req: CritiqueRequest | None = None) -> dict:
    req = req or CritiqueRequest()
    compare_against = None
    if req.compare_against_image_ref is not None and req.compare_against_score is not None:
        compare_against = {"image_ref": req.compare_against_image_ref, "score": req.compare_against_score}
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
    return {
        "total": preference_store.count(),
        "verifier_stack": preference_store.count(source="verifier_stack"),
        "gemma_critique": preference_store.count(source="gemma_critique"),
        "uicrit_seed": preference_store.count(source="uicrit_seed"),
    }


@app.post("/preference-pairs/export")
async def preference_pairs_export(source: str | None = None) -> dict:
    from krisna_training.dpo.dpo_dataset_export import export_jsonl

    output_path = f"krisna_dpo_export{'_' + source if source else ''}.jsonl"
    count = export_jsonl(preference_store, output_path, source=source)
    return {"written": count, "path": output_path}


@app.get("/orchestrator/status")
async def orchestrator_status() -> dict:
    return {
        "residency_state": orchestrator.state.state.value,
        "conversation_available": orchestrator.conversation_available(),
        "vram": orchestrator.ledger.snapshot(),
        # New: system-RAM ledger — always present (never None), since
        # it's a well-defined no-op snapshot in full-VRAM mode (every
        # resident spec has ram_gb=0.0) rather than something that only
        # exists conditionally on low_vram — a caller checking this
        # field doesn't need to know which mode is active first.
        "ram": orchestrator.ram_ledger.snapshot(),
        "low_vram_mode": orchestrator.low_vram,
        "history_len": len(orchestrator.state.history),
    }
