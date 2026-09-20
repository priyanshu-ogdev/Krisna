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
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure monorepo packages are importable when running without editable install
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in (_REPO_ROOT / "inference" / "src", _REPO_ROOT / "training" / "src"):
    if _pkg.exists() and str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

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
    from krisna_inference.common.hardware import (
        HardwareIncompatibleError,
        check_gpu_cuda_match,
        check_inference_config,
        format_diagnostic_report,
    )

    _low_vram = os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"

    # Fail-fast preflight verification: real backends require a compatible NVIDIA GPU + CUDA
    hw_status = check_gpu_cuda_match(require_gpu=False, low_vram=_low_vram)
    cfg_status = check_inference_config(require_real=True)
    if not hw_status.is_compatible or not cfg_status.is_ready:
        report = format_diagnostic_report(hw_status, cfg_status)
        log.error("hardware_preflight_failed", extra={"report": report})
        raise HardwareIncompatibleError(report)

    from krisna_inference.backends.factory import real_backend_factory

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
    from krisna_inference.backends.blob_store_singleton import get_blob_store
    from krisna_inference.backends.sketch_handoff import make_vq_decode_handoff
    from krisna_training.sketch.vq_tokenizer import VQTokenizer

    _vqgan_checkpoint = os.environ.get("KRISNA_VQGAN_CHECKPOINT")
    _vqgan_config = os.environ.get("KRISNA_VQGAN_CONFIG")
    # Grid dims must match whichever Sketch checkpoint is actually loaded
    # (KRISNA_SKETCH_CHECKPOINT) — Stage 1 is 16x16 (256px), Stage 2 is
    # 32x32 (512px). Explicit env vars rather than reading the loaded
    # SketchBackend's own config: that model loads asynchronously, after
    # this module-level setup runs.
    _sketch_grid_h = int(os.environ.get("KRISNA_SKETCH_GRID_H", "16"))
    _sketch_grid_w = int(os.environ.get("KRISNA_SKETCH_GRID_W", "16"))

    if _vqgan_checkpoint and _vqgan_config:
        _vq_tokenizer = VQTokenizer(checkpoint_path=_vqgan_checkpoint, config_path=_vqgan_config)
        _handoff_hook = make_vq_decode_handoff(_vq_tokenizer, _sketch_grid_h, _sketch_grid_w)
    else:
        log.warning(
            "vqgan_checkpoint_not_configured",
            extra={
                "note": (
                    "KRISNA_VQGAN_CHECKPOINT/KRISNA_VQGAN_CONFIG not set — "
                    "Finalize will fail loudly (not silently) on first use "
                    "rather than passing a token-grid blob ref to Polish as "
                    "if it were a decoded image."
                )
            },
        )

        def _handoff_hook(vq_tokens_ref: str) -> str:
            raise RuntimeError(
                "Cannot finalize: KRISNA_VQGAN_CHECKPOINT/KRISNA_VQGAN_CONFIG "
                "are not configured, so sketch tokens can't be decoded to a "
                "pixel image before handing off to the Polish tier. Set both "
                "env vars to the VQGAN (boris/vqgan_f16_16384) checkpoint/"
                "config used by Sketch-tier training."
            )
else:
    orchestrator = SwapOrchestrator()  # MockBackend — safe default, no GPU/weights needed
    _verifier_stack = None
    _handoff_hook = None  # flows.finalize()'s identity-passthrough default is fine for MockBackend — its polish backends don't call PIL.Image.open() on the ref at all.

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
@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "residency_state": orchestrator.state.state.value,
        "vram": orchestrator.ledger.snapshot(),
    }


@app.get("/hardware")
async def hardware_status():
    from krisna_inference.common.hardware import check_gpu_cuda_match, check_inference_config

    _low_vram = os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"
    hw = check_gpu_cuda_match(require_gpu=False, low_vram=_low_vram)
    cfg = check_inference_config(require_real=os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1")
    return {"hardware": hw.to_dict(), "config": cfg.to_dict()}


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

    from krisna_inference.backends.blob_store_singleton import get_blob_store

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
            handoff_hook=_handoff_hook,
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
    from krisna_inference.orchestrator.vram_budget import probe_real_ram, probe_real_vram

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


def main() -> None:
    """CLI entry point for running the Krisna Inference FastAPI service."""
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="krisna-inference",
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
        "krisna_inference.orchestrator.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

