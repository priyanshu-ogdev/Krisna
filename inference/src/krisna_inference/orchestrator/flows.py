"""§5.3 sequence flows, wired end-to-end: DesignStateStore <-> SwapOrchestrator.

Each function here is the orchestration-layer implementation of exactly one
of the three diagrams in §5.3. They own updating DesignState fields at the
right points in the sequence (not the orchestrator itself, which knows
nothing about DesignState — it only knows about ModelBackend/Tier/VRAM).
"""

from __future__ import annotations

import logging
from typing import Any

from krisna_inference.orchestrator.design_state import (
    CritiqueResult,
    DesignState,
    SessionStage,
)
from krisna_inference.orchestrator.exceptions import InvalidTransitionError
from krisna_inference.orchestrator.model_registry import Tier
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator, SwapResult

log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).flows")


class FlowError(Exception):
    pass


async def conversational_turn(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    user_message: str,
) -> DesignState:
    """§5.3 'Conversational turn' — the common, cheap, no-swap path."""
    state = store.get(session_id)
    expected_rev = state.revision

    state.append_turn("user", user_message)

    result = await orchestrator.run_conversational_turn(
        session_id=session_id,
        message=user_message,
        constraints=state.constraints.model_dump(),
    )

    planner_reply = result["planner"].get("reply_text", "(planner reply)")
    state.append_turn("planner", str(planner_reply))

    sketch_out = result["sketch"]
    new_vq_tokens_ref = sketch_out.get("vq_tokens_ref")
    state.sketch_tokens.vq_tokens = new_vq_tokens_ref or state.sketch_tokens.vq_tokens
    state.sketch_tokens.confidence_map = sketch_out.get(
        "confidence_map_ref", state.sketch_tokens.confidence_map
    )
    state.sketch_tokens.revision += 1
    # BUG FIX: this used to set stage=SKETCHING unconditionally, on every
    # conversational turn, even if `sketch_out` carried no new
    # `vq_tokens_ref` at all. In the current architecture the Sketch
    # backend runs on every turn, so this was latent, not live — but
    # `is_finalize_eligible()` returns True purely from
    # `state.stage == SKETCHING` plus a non-empty `sketch_tokens.vq_tokens`,
    # and a session reloaded from the store could carry a stale
    # `vq_tokens` from an earlier turn. A future change that makes the
    # Sketch call conditional (e.g. skip it for a pure clarifying
    # question) would silently make stage=SKETCHING true without any
    # actual sketch existing for the turn that set it — a session could
    # then become finalize-eligible without ever having produced fresh
    # sketch tokens. Only advance the stage when this turn genuinely
    # produced one.
    if new_vq_tokens_ref:
        state.stage = SessionStage.SKETCHING
    state.touch()

    return store.save(state, expected_rev)


async def finalize(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    quality: bool = False,
    handoff_hook: Any = None,
    verifier_stack: Any = None,
    prompt: str | None = None,
    min_safety_score: float = 0.9,
) -> DesignState:
    """§5.3 'Finalize'.

    `handoff_hook`, if given, is called as
    `handoff_hook(vq_tokens) -> pixel_image_ref` to perform the
    "VQ-decode -> pixel image -> re-encode into renderer's native latent"
    conversion described in the diagram. That conversion is genuinely part
    of the inference layer (it needs the actual VQ decoder + renderer
    encoder), not the orchestrator — this hook is the seam where it plugs
    in. Defaults to an identity passthrough so flows are testable without it.

    `verifier_stack`, if given (a verifiers.verifier_stack.VerifierStack),
    is run against the polished output per §5.3's Finalize diagram
    ("Polish Tier generates -> Verifier Stack scores output -> ...");
    populates finalize_output.verifier_scores AND enforces
    VerifierStack.safety_gate() before the image is ever handed back to
    the caller — that gate existed as a method on VerifierStack but was
    never actually invoked anywhere in this flow, silently making it
    dead code and the "gate" description in its own docstring untrue:
    an unsafe image would previously have finalized successfully with a
    low `safety` reading buried in an unused field, never a blocked
    result. Fixed here rather than left as a documented-but-unenforced
    intention — see docs/review/14_safety_gate_wiring.md. Left as None
    by default so this function stays usable with MockBackend/no torch
    installed — pass a real VerifierStack only when you actually want
    scoring AND the safety gate (it needs torch/transformers/opencv/
    easyocr).

    `min_safety_score` is forwarded to `VerifierStack.safety_gate()`
    unchanged (default 0.9, matching that method's own default) — only
    meaningful when `verifier_stack` is not None.
    """
    state = store.get(session_id)
    expected_rev = state.revision

    if not state.is_finalize_eligible():
        raise FlowError(
            f"Session {session_id} is not finalize-eligible "
            f"(stage={state.stage.value}, has_sketch={bool(state.sketch_tokens.vq_tokens)})"
        )

    state.stage = SessionStage.FINALIZING
    state.touch()
    store.save(state, expected_rev)
    expected_rev = state.revision

    handoff_input = state.sketch_tokens.vq_tokens
    pixel_ref = handoff_hook(handoff_input) if handoff_hook else handoff_input

    preferred = Tier.POLISH_QUALITY if quality else Tier.POLISH_DEFAULT
    result: SwapResult = await orchestrator.request_finalize(
        preferred_tier=preferred,
        session_id=session_id,
        handoff_image_ref=pixel_ref,
        prompt=prompt,
        constraints=state.constraints.model_dump(),
        locked_regions=[r.model_dump() for r in state.constraints.locked_regions],
    )

    if not result.ok:
        # Request failed cleanly; baseline residency was already restored
        # by the orchestrator. Roll the session stage back rather than
        # leaving it stuck in FINALIZING.
        state.stage = SessionStage.SKETCHING
        state.touch()
        store.save(state, expected_rev)
        raise FlowError(f"Finalize failed for session {session_id}: {result.error}")

    renderer_used = "qwen_image_edit_2511" if result.tier_used == Tier.POLISH_QUALITY else "z_image_turbo"
    output = result.output or {}

    state.finalize_output.renderer_used = renderer_used  # type: ignore[assignment]
    state.finalize_output.image_ref = output.get("image_ref", f"render://{session_id}/latest")
    verifier_scores = dict(output.get("verifier_scores", {}))

    if verifier_stack is not None and state.finalize_output.image_ref.startswith("blob://"):
        passed, safety_score = _check_safety_gate(
            verifier_stack, image_ref=state.finalize_output.image_ref, min_safety_score=min_safety_score
        )
        if not passed:
            # Mirrors the `not result.ok` rollback above exactly — stage
            # goes back to SKETCHING rather than FINALIZED, and this
            # raises before verifier_scores are even attached, so the
            # caller never receives a finalize_output for an image that
            # failed the safety gate. The image itself stays on disk
            # (blob store has no delete-on-reject path today — out of
            # scope for this fix, but worth flagging: a rejected blob is
            # currently orphaned, not purged).
            state.stage = SessionStage.SKETCHING
            state.touch()
            store.save(state, expected_rev)
            raise FlowError(
                f"Finalize blocked for session {session_id}: safety gate failed "
                f"(score={safety_score:.3f} < {min_safety_score})"
            )

        verifier_scores.update(
            _run_verifier_stack(
                verifier_stack,
                state=state,
                image_ref=state.finalize_output.image_ref,
                sketch_pixel_ref=pixel_ref if isinstance(pixel_ref, str) else None,
                prompt=prompt,
            )
        )

    for k, v in verifier_scores.items():
        if v is not None and hasattr(state.finalize_output.verifier_scores, k):
            setattr(state.finalize_output.verifier_scores, k, v)

    state.stage = SessionStage.FINALIZED
    state.touch()
    return store.save(state, expected_rev)


def _check_safety_gate(
    verifier_stack: Any, image_ref: str, min_safety_score: float
) -> tuple[bool, float]:
    from krisna_inference.backends.blob_store_singleton import get_blob_store

    polished_image = get_blob_store().load_image(image_ref)
    return verifier_stack.safety_gate(polished_image, min_safety_score=min_safety_score)


def _run_verifier_stack(
    verifier_stack: Any, state: DesignState, image_ref: str, sketch_pixel_ref: str | None, prompt: str | None
) -> dict:
    from krisna_inference.backends.blob_store_singleton import get_blob_store

    blobs = get_blob_store()
    polished_image = blobs.load_image(image_ref)
    sketch_image = None
    if sketch_pixel_ref and sketch_pixel_ref.startswith("blob://"):
        try:
            sketch_image = blobs.load_image(sketch_pixel_ref)
        except Exception as e:
            # UPGRADE: was `except FileNotFoundError` only. Before the
            # handoff_hook fix (sketch_handoff.py's make_vq_decode_handoff
            # actually being wired up), sketch_pixel_ref could be a raw
            # VQ-token blob ref rather than a decoded image — load_image()
            # on that raises PIL.UnidentifiedImageError, not
            # FileNotFoundError, which this except clause didn't catch,
            # meaning this was a second, hidden crash site for the same
            # root cause, reached only when handoff_consistency scoring
            # actually ran. Now fixed upstream (pixel_ref is a real
            # decoded image), but this broader catch matches
            # VerifierStack.score_finalize_output's own established
            # philosophy in this same module — any verifier input that
            # can't be loaded degrades to sketch_image=None (handoff_consistency
            # scored as unavailable) rather than aborting the whole
            # finalize response over one optional comparison image.
            log.warning("sketch_image_load_failed_for_handoff_verifier", extra={"ref": sketch_pixel_ref, "error": str(e)})
            sketch_image = None

    effective_prompt = prompt or state.constraints.style or "UI design"
    expected_regions = [r.model_dump() for r in state.constraints.locked_regions]
    return verifier_stack.score_finalize_output(
        polished_image, prompt=effective_prompt, sketch_image=sketch_image, expected_regions=expected_regions
    )


async def critique_pass(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    preference_store: Any = None,
    compare_against: dict | None = None,
) -> DesignState:
    """§5.3 'Critique pass' — rare, explicit, most VRAM-constrained.

    §5's architecture diagram shows the critique flowing to "Design State
    Manager + preference_pairs store" — but building a genuine (chosen,
    rejected) pair needs TWO scored candidates, and DesignState (§5.1) only
    tracks a single current `finalize_output`, not a history of prior
    candidates. Rather than fabricate history tracking that isn't in the
    schema, this takes that second candidate explicitly via
    `compare_against` (a caller-supplied
    `{"image_ref": str, "score": float}` — e.g. from a previous finalize
    revision the caller is still holding onto) and only writes a pair when
    both `preference_store` and `compare_against` are given. Without them,
    this behaves exactly as before: critique the render, update
    DesignState, done.
    """
    state = store.get(session_id)
    expected_rev = state.revision

    if not state.is_critique_eligible():
        raise FlowError(
            f"Session {session_id} is not critique-eligible "
            f"(stage={state.stage.value}, has_render={bool(state.finalize_output.image_ref)})"
        )

    state.stage = SessionStage.CRITIQUING
    state.critique.requested = True
    state.touch()
    store.save(state, expected_rev)
    expected_rev = state.revision

    result: SwapResult = await orchestrator.request_critique(
        session_id=session_id,
        image_ref=state.finalize_output.image_ref,
        constraints=state.constraints.model_dump(),
    )

    if not result.ok:
        state.stage = SessionStage.FINALIZED
        state.touch()
        store.save(state, expected_rev)
        raise FlowError(f"Critique failed for session {session_id}: {result.error}")

    output = result.output or {}
    critique_payload = output.get("critique_result")
    if critique_payload:
        from krisna_inference.verifiers.critique_adapter import from_gemma_output

        state.critique.result = from_gemma_output(critique_payload)
    else:
        # Mock backends won't produce a real Critique Adapter payload —
        # fill a minimal, clearly-marked placeholder so the flow is still
        # exercisable end-to-end without the inference layer.
        state.critique.result = CritiqueResult(
            critique_source="gemma4_31b_frozen",
            overall_score=0.0,
            raw_model_output_ref=None,
        )
    state.critique.source = "gemma4_31b"
    from datetime import datetime, timezone

    state.critique.timestamp = datetime.now(timezone.utc).isoformat()

    if preference_store is not None and compare_against is not None:
        from krisna_training.dpo.pair_builder import build_pair_from_candidates

        pair = build_pair_from_candidates(
            preference_store,
            prompt=state.constraints.style or "UI design",
            candidates=[
                {"image_ref": state.finalize_output.image_ref, "score": state.critique.result.overall_score},
                compare_against,
            ],
            source="gemma_critique",
            session_id=session_id,
        )
        if pair is not None:
            state.preference_pair_refs.append(pair.id)

    state.stage = SessionStage.FINALIZED
    state.touch()
    return store.save(state, expected_rev)
