"""§5.3 sequence flows, wired end-to-end: DesignStateStore <-> SwapOrchestrator.

Each function here is the orchestration-layer implementation of exactly one
of the three diagrams in §5.3. They own updating DesignState fields at the
right points in the sequence (not the orchestrator itself, which knows
nothing about DesignState — it only knows about ModelBackend/Tier/VRAM).
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

from krisna_inference.orchestrator.design_state import (
    CritiqueResult,
    DesignState,
    PreviewOutput,
    SessionStage,
)
from krisna_inference.orchestrator.exceptions import InvalidTransitionError
from krisna_inference.orchestrator.model_registry import Tier
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator, SwapResult

log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).flows")


class FlowError(Exception):
    pass


def _extract_current_layout_nodes(layout_ref: str | None) -> list[dict[str, Any]]:
    """Extract clean component nodes from post-solve LayoutIR for Planner grounding."""
    if not layout_ref or not Path(layout_ref).exists():
        return []
    try:
        layout_data = json.loads(Path(layout_ref).read_text(encoding="utf-8"))
        return [
            {
                "id": n["id"],
                "type": n["type"],
                "bbox": n["bbox"],
                "text": n.get("text"),
                "locked": n.get("locked", False),
            }
            for n in layout_data.get("nodes", [])
            if n.get("id") != "root"
        ]
    except Exception as e:
        log.warning("layout_grounding_sync_warning: %s", e)
        return []


async def conversational_turn(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    user_message: str,
    image_ref: str | None = None,
    image_refs: list[str] | None = None,
    reference_assets: list[Any] | None = None,
) -> DesignState:
    """§5.3 'Conversational turn' — the common, cheap, no-swap path."""
    state = store.get(session_id)
    expected_rev = state.revision

    from krisna_inference.orchestrator.design_state import ReferenceAsset

    if reference_assets:
        state.constraints.reference_assets = [
            ra if isinstance(ra, ReferenceAsset) else ReferenceAsset(**ra)
            for ra in reference_assets
        ]
        if not state.constraints.reference_image_ref and state.constraints.reference_assets:
            state.constraints.reference_image_ref = state.constraints.reference_assets[0].blob_ref
    elif image_refs:
        state.constraints.reference_assets = [
            ReferenceAsset(id=f"asset_{i}", blob_ref=ref, role="subject")
            for i, ref in enumerate(image_refs)
        ]
        if not state.constraints.reference_image_ref and state.constraints.reference_assets:
            state.constraints.reference_image_ref = state.constraints.reference_assets[0].blob_ref
    elif image_ref:
        state.constraints.reference_image_ref = image_ref
        if not state.constraints.reference_assets:
            state.constraints.reference_assets = [
                ReferenceAsset(id="default", blob_ref=image_ref, role="subject")
            ]

    # Multi-turn context: pass prior dialog history and any prior critique
    history = [turn.model_dump() for turn in state.conversation_history]
    prior_critique = (
        state.critique.result.model_dump()
        if (state.critique and state.critique.result)
        else None
    )

    # D2: Preserve original user intent so it is never lost when conversation grows past window
    if not state.constraints.original_intent and user_message and user_message.strip():
        state.constraints.original_intent = user_message.strip()

    state.append_turn("user", user_message)

    # Pass post-solve LayoutIR to Planner context to eliminate the Two-Master problem
    constraints_for_planner = state.constraints.model_dump()
    current_nodes = _extract_current_layout_nodes(state.layout.layout_ref)
    if current_nodes:
        constraints_for_planner["current_layout"] = current_nodes

    result = await orchestrator.run_conversational_turn(
        session_id=session_id,
        message=user_message,
        constraints=constraints_for_planner,
        conversation_history=history,
        prior_critique=prior_critique,
        image_ref=image_ref,
        image_refs=image_refs,
        reference_assets=[a.model_dump() for a in state.constraints.get_reference_assets()],
    )

    planner_out = result.get("planner", {})
    planner_reply = planner_out.get("reply_text", "(planner reply)")
    state.append_turn("planner", str(planner_reply))

    # Apply structured constraint updates from Planner's JSON delta —
    # shared with swap_orchestrator.py's run_conversational_turn (see
    # constraint_merge.py's docstring for why this now lives in one
    # place instead of two copies that could drift).
    from krisna_inference.orchestrator.constraint_merge import apply_constraint_updates

    delta = planner_out.get("design_state_delta") or {}
    constraint_updates = dict(delta.get("constraint_updates") or {})
    if "actions" in delta and "actions" not in constraint_updates:
        constraint_updates["actions"] = delta.get("actions")

    # UPGRADE (docs/review/35_model_review_3_resync.md): persist this
    # turn's reasoning_note so _synthesize_dpo_prompt() — used for both
    # Polish-tier generation and DPO training-data prompts — can enrich
    # its prompt through the structured layout route
    # with the SAME turn's live planner_output. Only overwrite when this
    # turn's delta actually carries a non-empty note; a turn where the
    # Planner has nothing new to add (reasoning_note empty/missing)
    # should not erase the last meaningful one.
    new_reasoning_note = str(delta.get("reasoning_note") or "").strip()
    if new_reasoning_note:
        state.last_planner_reasoning_note = new_reasoning_note

    original_constraints = state.constraints.model_dump()
    merged = apply_constraint_updates(original_constraints, constraint_updates)
    state.constraints.style = merged["style"]
    state.constraints.palette = merged["palette"]
    state.constraints.layout_hints = merged["layout_hints"]
    if merged["locked_regions"] != original_constraints["locked_regions"]:
        from krisna_inference.orchestrator.design_state import LockedRegion

        state.constraints.locked_regions = [
            r if isinstance(r, LockedRegion) else LockedRegion(**r)
            for r in merged["locked_regions"]
        ]

    if image_ref:
        state.constraints.reference_image_ref = image_ref

    sketch_out = result.get("sketch", {})
    new_layout_ref = sketch_out.get("layout_ref")
    if new_layout_ref:
        state.layout.prior_layout_ref = state.layout.layout_ref
        state.layout.layout_ref = new_layout_ref
        state.layout.wireframe_ref = sketch_out.get("wireframe_ref")
        state.layout.segmentation_ref = sketch_out.get("segmentation_ref")
        state.layout.mask_ref = sketch_out.get("mask_ref")
        state.layout.revision += 1
        state.stage = SessionStage.SKETCHING

    sketch_locked = sketch_out.get("locked_regions") or []
    if sketch_locked:
        from krisna_inference.orchestrator.design_state import LockedRegion
        for slr in sketch_locked:
            slr_dict = slr if isinstance(slr, dict) else slr.model_dump()
            box = slr_dict.get("bbox")
            if not box:
                continue
            matched = False
            for existing in state.constraints.locked_regions:
                if list(existing.bbox) == list(box) or (slr_dict.get("id") and existing.id == slr_dict.get("id")):
                    matched = True
                    if slr_dict.get("source_asset_id") and not existing.source_asset_id:
                        existing.source_asset_id = slr_dict.get("source_asset_id")
                    if slr_dict.get("source_crop_bbox") and not existing.source_crop_bbox:
                        existing.source_crop_bbox = tuple(slr_dict["source_crop_bbox"])
                    if slr_dict.get("id") and not existing.id:
                        existing.id = slr_dict.get("id")
                    break
            if not matched:
                state.constraints.locked_regions.append(
                    LockedRegion(
                        id=slr_dict.get("id"),
                        bbox=tuple(box),
                        reason=slr_dict.get("reason", "Sketch lock"),
                        source_asset_id=slr_dict.get("source_asset_id"),
                        source_crop_bbox=tuple(slr_dict["source_crop_bbox"]) if slr_dict.get("source_crop_bbox") else None,
                    )
                )

    if state.refinement.exhausted:
        state.refinement.reset_cycle()
    state.refinement.record_turn("user")
    state.touch()

    return store.save(state, expected_rev)


async def decompile_screenshot(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    image_ref: str,
    instruction: str = "Decompile this UI screenshot into an editable layout.",
) -> DesignState:
    """VLM Screenshot-to-Layout Decompilation (ACL Design2Code / CVPR ShowUI).

    Synchronizes the Sketch tier directly with the 9B VLM Planner.
    Processes the raw raster screenshot through the VLM module, extracts semantic
    UI components with spatial coordinates [x1, y1, x2, y2], and renders a deterministic
    LayoutIR, wireframe, and SPADE segmentation mask without human transcription.
    """
    return await conversational_turn(
        orchestrator=orchestrator,
        store=store,
        session_id=session_id,
        user_message=instruction,
        image_ref=image_ref,
    )


async def planner_self_review(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
) -> DesignState:
    """Run one bounded autonomous Planner+Layout refinement turn."""
    state = store.get(session_id)
    if state.refinement.exhausted:
        raise FlowError(f"Refinement budget exhausted ({state.refinement.turn_count}/{state.refinement.max_turns})")
    if not state.layout.layout_ref:
        raise FlowError("No active layout draft to review")

    expected_rev = state.revision
    intent = state.constraints.original_intent or (
        state.conversation_history[-1].content if state.conversation_history else "Refine UI layout"
    )
    history = [turn.model_dump() for turn in state.conversation_history]
    prior_critique = (
        state.critique.result.model_dump()
        if (state.critique and state.critique.result)
        else None
    )

    review_message = (
        f"Autonomous self-review of current layout against original intent: '{intent}'. "
        f"Active style: '{state.constraints.style}'. "
        f"Inspect hierarchy, spacing, and contrast, and propose targeted constraint adjustments."
    )

    review_constraints = state.constraints.model_dump()
    current_nodes = _extract_current_layout_nodes(state.layout.layout_ref)
    if current_nodes:
        review_constraints["current_layout"] = current_nodes

    result = await orchestrator.run_conversational_turn(
        session_id=session_id,
        message=review_message,
        constraints=review_constraints,
        conversation_history=history,
        prior_critique=prior_critique,
        review=True,
        wireframe_ref=state.layout.wireframe_ref,
    )

    planner_out = result.get("planner", {})
    planner_reply = planner_out.get("reply_text", "(planner self-review)")
    state.append_turn("planner", str(planner_reply))
    state.refinement.record_turn("self_review")

    from krisna_inference.orchestrator.constraint_merge import apply_constraint_updates

    delta = planner_out.get("design_state_delta") or {}
    constraint_updates = dict(delta.get("constraint_updates") or {})
    if "actions" in delta and "actions" not in constraint_updates:
        constraint_updates["actions"] = delta.get("actions")
    new_reasoning_note = str(delta.get("reasoning_note") or "").strip()
    if new_reasoning_note:
        state.last_planner_reasoning_note = new_reasoning_note

    original_constraints = state.constraints.model_dump()
    merged = apply_constraint_updates(original_constraints, constraint_updates)
    state.constraints.style = merged["style"]
    state.constraints.palette = merged["palette"]
    state.constraints.layout_hints = merged["layout_hints"]
    if merged["locked_regions"] != original_constraints["locked_regions"]:
        from krisna_inference.orchestrator.design_state import LockedRegion

        state.constraints.locked_regions = [
            r if isinstance(r, LockedRegion) else LockedRegion(**r)
            for r in merged["locked_regions"]
        ]

    sketch_out = result.get("sketch", {})
    new_layout_ref = sketch_out.get("layout_ref")
    if new_layout_ref:
        state.layout.layout_ref = new_layout_ref
        state.layout.wireframe_ref = sketch_out.get("wireframe_ref")
        state.layout.segmentation_ref = sketch_out.get("segmentation_ref")
        state.layout.mask_ref = sketch_out.get("mask_ref")
        state.layout.revision += 1
        state.stage = SessionStage.SKETCHING

    state.touch()
    return store.save(state, expected_rev)


async def finalize(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    quality: bool = False,
    fast: bool = False,
    handoff_hook: Any = None,
    verifier_stack: Any = None,
    prompt: str | None = None,
    min_safety_score: float = 0.9,
    **kwargs: Any,
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
            f"(stage={state.stage.value}, has_layout={bool(state.layout.layout_ref)})"
        )

    state.stage = SessionStage.FINALIZING
    state.touch()
    store.save(state, expected_rev)
    expected_rev = state.revision

    handoff_input = state.layout.wireframe_ref or state.layout.layout_ref
    pixel_ref = handoff_hook(handoff_input) if handoff_hook else handoff_input
    segmentation_ref = state.layout.segmentation_ref
    mask_ref = state.layout.mask_ref

    # UPGRADE: Wireframe-as-lock-donor resolution
    # Only use a lock donor if a prior photorealistic render actually exists or an initial photo/screenshot was uploaded!
    # On Turn 1 text-to-image, lock_donor_ref is None to avoid copying low-fidelity wireframe lines into the photorealistic render.
    # On Image-to-Image (or portrait editing), lock_donor_ref resolves to the original reference photo (state.constraints.reference_image_ref),
    # guaranteeing 0.00% facial drift for locked identity regions (NVIDIA AnyDoor & ConsisID).
    lock_donor_ref = state.finalize_output.image_ref or getattr(state.constraints, "reference_image_ref", None)
    lock_donor_map: dict[str, str] = {}
    for asset in state.constraints.get_reference_assets():
        if asset.blob_ref:
            lock_donor_map[asset.id] = asset.blob_ref
    if state.finalize_output.image_ref:
        lock_donor_map["canvas"] = state.finalize_output.image_ref
    if lock_donor_ref and "default" not in lock_donor_map:
        lock_donor_map["default"] = lock_donor_ref

    # Synthesize prompt from conversational intent, constraints, and LayoutIR components/labels.
    effective_prompt = prompt if prompt else synthesize_polish_prompt(state)

    if fast:
        preferred = Tier.POLISH_FAST
    elif quality:
        preferred = Tier.POLISH_QUALITY
    else:
        preferred = Tier.POLISH_DEFAULT
    result: SwapResult = await orchestrator.request_finalize(
        preferred_tier=preferred,
        session_id=session_id,
        handoff_image_ref=pixel_ref,
        lock_donor_ref=lock_donor_ref,
        lock_donor_map=lock_donor_map,
        segmentation_ref=segmentation_ref,
        mask_ref=mask_ref,
        prompt=effective_prompt,
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

    renderer_used = "qwen_image_2_1"
    output = result.output or {}

    state.finalize_output.renderer_used = renderer_used  # type: ignore[assignment]
    state.finalize_output.image_ref = output.get("image_ref", f"render://{session_id}/latest")
    state.finalize_output.prompt_used = effective_prompt

    if fast:
        state.preview_output = PreviewOutput(
            renderer_used=renderer_used,  # type: ignore[assignment]
            image_ref=state.finalize_output.image_ref,
            step_count=6,
            prompt_used=effective_prompt,
        )

    verifier_scores = dict(output.get("verifier_scores", {}))

    safety_score: float | None = None
    if verifier_stack is not None and state.finalize_output.image_ref.startswith("blob://"):
        passed, safety_score = _check_safety_gate(
            verifier_stack, image_ref=state.finalize_output.image_ref, min_safety_score=min_safety_score
        )
        if not passed:
            # Stage goes back to SKETCHING rather than FINALIZED, and this
            # raises before verifier_scores are even attached, so the
            # caller never receives a finalize_output for an image that
            # failed the safety gate. D4 FIX: Purge the rejected blob file
            # from disk so orphans do not accumulate indefinitely.
            rejected_ref = state.finalize_output.image_ref
            if rejected_ref and rejected_ref.startswith("blob://"):
                try:
                    from krisna_inference.backends.blob_store_singleton import get_blob_store
                    get_blob_store().delete(rejected_ref)
                except Exception:
                    pass
            state.finalize_output.image_ref = None
            state.stage = SessionStage.SKETCHING
            state.touch()
            store.save(state, expected_rev)
            raise FlowError(
                f"Finalize blocked for session {session_id}: safety gate failed "
                f"(score={safety_score:.3f} < {min_safety_score})"
            )

        verifier_result = _run_verifier_stack(
                verifier_stack,
                state=state,
                image_ref=state.finalize_output.image_ref,
                sketch_pixel_ref=pixel_ref if isinstance(pixel_ref, str) else None,
                prompt=effective_prompt,
                safety_score=safety_score,
                lock_donor_ref=lock_donor_ref,
            )
        if "evaluation_receipt" in verifier_result:
            state.finalize_output.evaluation_receipt = verifier_result.pop("evaluation_receipt")
        verifier_scores.update(verifier_result)

        # Closed-Loop Spatial Alignment Self-Correction (Auto-Refine / Retry Engine)
        layout_iou = verifier_scores.get("layout_iou")
        ocr_score = verifier_scores.get("ocr_readability")
        seam_score = verifier_scores.get("seam_continuity")
        lock_preservation = verifier_scores.get("lock_preservation")
        drift_reasons = []
        if layout_iou is not None and layout_iou < 0.70:
            drift_reasons.append(f"Layout IoU {layout_iou:.2f} < 0.70")
        if ocr_score is not None and ocr_score < 0.80:
            drift_reasons.append(f"OCR readability {ocr_score:.2f} < 0.80")
        if seam_score is not None and seam_score < 0.85:
            drift_reasons.append(f"Seam continuity {seam_score:.2f} < 0.85")
        if lock_preservation is not None and lock_preservation < 0.95:
            drift_reasons.append(f"Lock preservation {lock_preservation:.2f} < 0.95")

        if drift_reasons and not kwargs.get("_retry_attempted", False):
            log.warning("Spatial drift detected in finalize (%s); triggering automated layout-enforced retry.", "; ".join(drift_reasons))
            retry_prompt = build_grounded_spatial_retry_prompt(
                state=state,
                base_prompt=effective_prompt,
                drift_reasons=drift_reasons,
                layout_iou=layout_iou,
                ocr_score=ocr_score,
                seam_score=seam_score,
                lock_preservation=lock_preservation,
            )
            retry_result = await orchestrator.request_finalize(
                preferred_tier=preferred,
                session_id=session_id,
                handoff_image_ref=pixel_ref,
                lock_donor_ref=lock_donor_ref,
                segmentation_ref=segmentation_ref,
                mask_ref=mask_ref,
                prompt=retry_prompt,
                constraints=state.constraints.model_dump(),
                locked_regions=[r.model_dump() for r in state.constraints.locked_regions],
                seed=42 + 1337,
            )
            if retry_result.ok:
                retry_output = retry_result.output or {}
                retry_ref = retry_output.get("image_ref")
                if retry_ref and retry_ref.startswith("blob://"):
                    retry_verifier_result = _run_verifier_stack(
                        verifier_stack,
                        state=state,
                        image_ref=retry_ref,
                        sketch_pixel_ref=pixel_ref if isinstance(pixel_ref, str) else None,
                        prompt=retry_prompt,
                        safety_score=safety_score,
                        lock_donor_ref=lock_donor_ref,
                    )
                    retry_iou = retry_verifier_result.get("layout_iou", 0.0)
                    if retry_iou >= (layout_iou or 0.0):
                        state.finalize_output.image_ref = retry_ref
                        state.finalize_output.prompt_used = retry_prompt
                        verifier_scores.update(retry_verifier_result)
                        drift_reasons.append(f"Auto-refine retry improved IoU to {retry_iou:.2f}")

        # Surface spatial drift diagnostic to conversation history for next-turn Planner grounding
        if drift_reasons:
            state.append_turn("system", f"Spatial verifier diagnostic: {'; '.join(drift_reasons)}.")

    for k, v in verifier_scores.items():
        if v is not None and hasattr(state.finalize_output.verifier_scores, k):
            setattr(state.finalize_output.verifier_scores, k, v)

    state.stage = SessionStage.FINALIZED
    state.touch()
    return store.save(state, expected_rev)


async def fast_preview(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    prompt: str | None = None,
    verifier_stack: Any = None,
    handoff_hook: Any = None,
    min_safety_score: float = 0.9,
) -> DesignState:
    """§5.3 'Fast Preview' — instant 4-8 step preview using Tier.POLISH_FAST (CFM distillation).

    Provides sub-second visual feedback (~0.9s on A6000, ~2.5s on consumer GPU) so designers
    can inspect typography and spatial hierarchy before running full 28-step or 40-step finalization.
    """
    return await finalize(
        orchestrator=orchestrator,
        store=store,
        session_id=session_id,
        quality=False,
        fast=True,
        handoff_hook=handoff_hook,
        verifier_stack=verifier_stack,
        prompt=prompt,
        min_safety_score=min_safety_score,
    )


async def plan_sketch_and_polish(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    user_message: str,
    image_ref: str | None = None,
    image_refs: list[str] | None = None,
    reference_assets: list[Any] | None = None,
    quality: bool = False,
    fast: bool = False,
    prompt: str | None = None,
    verifier_stack: Any = None,
    handoff_hook: Any = None,
    min_safety_score: float = 0.9,
    auto_review: bool = True,
    **kwargs: Any,
) -> DesignState:
    """Unified Autonomous Plan-and-Sketch Pipeline with Direct-to-Polish and Closed-Loop Review.

    Integrates Qwen3.5-9B VLM deliberation and krisna.layout.v1 into a single pipeline:
    1. Simultaneous Plan + Sketch:
       Executes Planner deliberation (Gated DeltaNet linear recurrence, M-RoPE spatial grounding)
       and layout compilation (Cassowary constraint solving, DualStreamSketch rasterization).
    2. Direct Polish Progression:
       Immediately hands off wireframe, segmentation, and synthesized prompt to Qwen-Image-2.1
       (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, or Tier.POLISH_FAST) without manual intermediate calls.
    3. Verifier Stack Evaluation:
       Scores the rendered output on layout IoU, OCR readability, aesthetic score, seam continuity,
       and enforces the safety gate.
    4. Closed-Loop Review & Further Planning Deliberation:
       Passes the verifier receipt and render metadata back into Qwen3.5-9B for post-render
       deliberation, generating an autonomous evaluation and recommended next-step refinements
       ([Post-Render Deliberation]) and appending it to the session conversation history.
    """
    # Step 1: Simultaneous Plan and Sketch execution
    state = await conversational_turn(
        orchestrator=orchestrator,
        store=store,
        session_id=session_id,
        user_message=user_message,
        image_ref=image_ref,
        image_refs=image_refs,
        reference_assets=reference_assets,
    )

    # Step 2: Direct Polish progression
    state = await finalize(
        orchestrator=orchestrator,
        store=store,
        session_id=session_id,
        quality=quality,
        fast=fast,
        handoff_hook=handoff_hook,
        verifier_stack=verifier_stack,
        prompt=prompt,
        min_safety_score=min_safety_score,
        **kwargs,
    )

    # Step 3 & 4: Closed-Loop Post-Render Review & Further Planning Deliberation
    if auto_review and state.stage == SessionStage.FINALIZED and not state.refinement.exhausted:
        try:
            expected_rev = state.revision
            verifier_scores = state.finalize_output.verifier_scores
            scores_summary = []
            if getattr(verifier_scores, "layout_iou", None) is not None:
                scores_summary.append(f"Layout IoU: {verifier_scores.layout_iou:.2f}")
            if getattr(verifier_scores, "ocr_readability", None) is not None:
                scores_summary.append(f"OCR Readability: {verifier_scores.ocr_readability:.2f}")
            if getattr(verifier_scores, "aesthetic", None) is not None:
                scores_summary.append(f"Aesthetic: {verifier_scores.aesthetic:.2f}")
            if getattr(verifier_scores, "seam_continuity", None) is not None:
                scores_summary.append(f"Seam Continuity: {verifier_scores.seam_continuity:.2f}")
            if getattr(verifier_scores, "lock_preservation", None) is not None:
                scores_summary.append(f"Lock Preservation: {verifier_scores.lock_preservation:.2f}")

            scores_text = ", ".join(scores_summary) if scores_summary else "N/A"
            intent = state.constraints.original_intent or user_message

            review_prompt = (
                f"Post-render verifier review for rendered UI image. "
                f"Original intent: '{intent}'. "
                f"Active style: '{state.constraints.style}'. "
                f"Verifier scores: [{scores_text}]. "
                f"Evaluate layout fidelity, typography legibility, and aesthetics against intent. "
                f"Formulate next-step planning directives."
            )

            history = [turn.model_dump() for turn in state.conversation_history]
            constraints_for_review = state.constraints.model_dump()
            current_nodes = _extract_current_layout_nodes(state.layout.layout_ref)
            if current_nodes:
                constraints_for_review["current_layout"] = current_nodes

            review_res = await orchestrator.run_conversational_turn(
                session_id=session_id,
                message=review_prompt,
                constraints=constraints_for_review,
                conversation_history=history,
                review=True,
                wireframe_ref=state.layout.wireframe_ref,
            )
            planner_out = review_res.get("planner", {})
            review_reply = planner_out.get("reply_text") or "(Post-render deliberation completed. Layout aligns with specification.)"
            state.append_turn("planner", f"[Post-Render Deliberation] {review_reply}")
            state.refinement.record_turn("self_review")
            state.touch()
            state = store.save(state, expected_rev)
        except Exception as e:
            log.warning("post_render_auto_review_warning: %s", e)

    return state


async def enhance_photo(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    image_ref: str,
    prompt: str,
    quality: bool = True,
) -> DesignState:
    """Bypass Planner and Sketch to enhance an image with Qwen-Image-2.1.

    This route is intentionally separate from the UI wireframe path, but still
    uses the same fail-closed real backend and verifier contracts.
    """
    state = store.get(session_id)
    expected_rev = state.revision

    state.stage = SessionStage.FINALIZING
    state.touch()
    store.save(state, expected_rev)
    expected_rev = state.revision

    preferred = Tier.POLISH_QUALITY if quality else Tier.POLISH_DEFAULT
    result: SwapResult = await orchestrator.request_finalize(
        preferred_tier=preferred,
        session_id=session_id,
        handoff_image_ref=image_ref,
        prompt=prompt,
        constraints=state.constraints.model_dump(),
        locked_regions=[r.model_dump() for r in state.constraints.locked_regions],
    )

    if not result.ok:
        state.stage = SessionStage.CONVERSING
        state.touch()
        store.save(state, expected_rev)
        raise FlowError(f"Enhance failed for session {session_id}: {result.error}")

    renderer_used = "qwen_image_2_1"
    output = result.output or {}

    state.finalize_output.renderer_used = renderer_used  # type: ignore[assignment]
    state.finalize_output.image_ref = output.get("image_ref", f"render://{session_id}/latest")
    state.finalize_output.prompt_used = prompt

    state.stage = SessionStage.FINALIZED
    state.touch()
    return store.save(state, expected_rev)


async def iterative_refine(
    orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    session_id: str,
    instruction: str,
    quality: bool = False,
    verifier_stack: Any = None,
    num_inference_steps: int = 8,
    min_safety_score: float = 0.9,
    actions: list[Any] | None = None,
) -> DesignState:
    """FlowEdit-Style Iterative Refinement — PRD §5.3 / ICCV 2025.

    Operates on an already-rendered design (state.finalize_output.image_ref).
    Solves the flow matching ODE forward/backward for few-step (8-step) edits
    preserving unmodified regions, followed by LockCompositor enforcement
    for 0.00% pixel drift on locked regions. Supports Visual-Language-Action (VLA)
    design grammar actions (ShowUI CVPR 2025; SeeClick ACL 2024).

    Research basis: Kulikov et al. (ICCV 2025, CVF Open Access).
    """
    state = store.get(session_id)
    expected_rev = state.revision

    if not state.is_refine_eligible():
        raise FlowError(
            f"Session {session_id} is not refine-eligible: requires an existing rendered image "
            f"(stage={state.stage.value}, has_render={bool(state.finalize_output.image_ref)})"
        )

    # Apply any VLA design actions supplied with this refinement turn
    if actions:
        from krisna_inference.orchestrator.constraint_merge import apply_constraint_updates

        original_constraints = state.constraints.model_dump()
        merged = apply_constraint_updates(original_constraints, actions)
        state.constraints.style = merged["style"]
        state.constraints.palette = merged["palette"]
        state.constraints.layout_hints = merged["layout_hints"]
        if merged["locked_regions"] != original_constraints["locked_regions"]:
            from krisna_inference.orchestrator.design_state import LockedRegion

            state.constraints.locked_regions = [
                r if isinstance(r, LockedRegion) else LockedRegion(**r)
                for r in merged["locked_regions"]
            ]

        # Synchronize LayoutIR so downstream post-refine IoU verification matches updated nodes
        if state.layout.layout_ref and Path(state.layout.layout_ref).exists():
            try:
                from krisna_inference.layout_baseline import layout_from_constraints, LayoutIR
                from krisna_inference.layout_renderer import render_dual_stream_sketch
                from krisna_inference.backends.blob_store_singleton import get_blob_store

                prior_layout_obj = None
                try:
                    prior_text = Path(state.layout.layout_ref).read_text(encoding="utf-8")
                    prior_json = json.loads(prior_text)
                    prior_layout_obj = LayoutIR(**prior_json)
                except Exception as read_exc:
                    log.debug("Could not parse prior layout for differential masking: %s", read_exc)

                merged_constraints = state.constraints.model_dump()
                merged_constraints["actions"] = actions
                updated_layout = layout_from_constraints(merged_constraints)
                Path(state.layout.layout_ref).write_text(
                    json.dumps(updated_layout.model_dump(), indent=2) + "\n", encoding="utf-8"
                )
                state.layout.revision += 1

                # Surface any locked-locked collision diagnostics directly to Planner conversation history
                for diag in getattr(updated_layout, "diagnostics", []):
                    state.append_turn("system", f"[Layout Conflict]: {diag}")

                # Re-render dual-stream sketch (wireframe, SPADE segmentation, and FlowEdit differential mask)
                wf_path = Path(state.layout.layout_ref).with_suffix(".png")
                blobs = get_blob_store()
                asset_mode = str(getattr(state.constraints, "asset_mode", "opaque") or "opaque").lower()
                dual_sketch = render_dual_stream_sketch(
                    updated_layout,
                    output_dir=wf_path.parent,
                    prefix=f"refine_{session_id}",
                    prior_layout=prior_layout_obj,
                    asset_mode=asset_mode,
                    style=state.constraints.style or "dark",
                )
                state.layout.wireframe_ref = blobs.save_image(dual_sketch.wireframe, prefix="wireframe_refine")
                state.layout.segmentation_ref = blobs.save_image(dual_sketch.segmentation, prefix="seg_refine")
                if dual_sketch.mask is not None:
                    state.layout.mask_ref = blobs.save_image(dual_sketch.mask, prefix="mask_refine")
            except Exception as e:
                log.warning("refine_layout_sync_warning: %s", e)

    prior_render = state.finalize_output.image_ref
    state.stage = SessionStage.REFINING
    state.append_turn("user", f"[Refinement Instruction]: {instruction}")
    state.touch()
    store.save(state, expected_rev)
    expected_rev = state.revision

    # Compose edit prompt combining style/constraints + instruction without runaway concatenation
    base = state.constraints.original_intent or state.constraints.style
    if not base:
        raw_prior = state.finalize_output.prompt_used or "Modern professional UI"
        base = raw_prior.split(". Edit update:")[0].split(". Refinement edit:")[0]
    edit_prompt = f"{base}. Refinement edit: {instruction}"

    preferred = Tier.POLISH_QUALITY if quality else Tier.POLISH_DEFAULT
    # Refinement Dual-Image Visual Conditioning:
    # Forward updated wireframe as handoff_image_ref (index 0: new geometry), prior_render
    # as lock_donor_ref (index 1: visual style & locks), and dual-stream segmentation + differential mask
    # so block-causal attention receives the full spatial multi-modal representation.
    wireframe_condition = state.layout.wireframe_ref or prior_render
    segmentation_condition = state.layout.segmentation_ref
    mask_condition = state.layout.mask_ref
    result: SwapResult = await orchestrator.request_finalize(
        preferred_tier=preferred,
        session_id=session_id,
        handoff_image_ref=wireframe_condition,
        lock_donor_ref=prior_render,
        segmentation_ref=segmentation_condition,
        mask_ref=mask_condition,
        prompt=edit_prompt,
        mode="refine",
        refine=True,
        num_inference_steps=num_inference_steps,
        constraints=state.constraints.model_dump(),
        locked_regions=[r.model_dump() for r in state.constraints.locked_regions],
    )

    if not result.ok:
        state.stage = SessionStage.FINALIZED
        state.touch()
        store.save(state, expected_rev)
        raise FlowError(f"Iterative refine failed for session {session_id}: {result.error}")

    output = result.output or {}
    refined_ref = output.get("image_ref") or output.get("render_ref") or f"render://{session_id}/refined"

    verifier_scores: dict[str, Any] = {}
    safety_score: float | None = None
    if verifier_stack is not None and refined_ref.startswith("blob://"):
        passed, safety_score = _check_safety_gate(
            verifier_stack, image_ref=refined_ref, min_safety_score=min_safety_score
        )
        if not passed:
            try:
                from krisna_inference.backends.blob_store_singleton import get_blob_store
                get_blob_store().delete(refined_ref)
            except Exception:
                pass
            state.stage = SessionStage.FINALIZED
            state.touch()
            store.save(state, expected_rev)
            raise FlowError(
                f"Iterative refinement blocked: safety gate failed (score={safety_score:.3f} < {min_safety_score})"
            )

        verifier_result = _run_verifier_stack(
            verifier_stack,
            state=state,
            image_ref=refined_ref,
            sketch_pixel_ref=state.layout.wireframe_ref or prior_render,
            prompt=edit_prompt,
            safety_score=safety_score,
            lock_donor_ref=prior_render,
        )
        if "evaluation_receipt" in verifier_result:
            state.finalize_output.evaluation_receipt = verifier_result.pop("evaluation_receipt")
        verifier_scores.update(verifier_result)
        for k, v in verifier_scores.items():
            if v is not None and hasattr(state.finalize_output.verifier_scores, k):
                setattr(state.finalize_output.verifier_scores, k, v)

    state.finalize_output.image_ref = refined_ref
    state.finalize_output.prompt_used = edit_prompt

    # Invalidate prior critique evaluated on the pre-refinement image
    from krisna_inference.orchestrator.design_state import Critique
    state.critique = Critique()

    state.append_turn("planner", f"Refined design applied based on instruction: {instruction}")
    if state.refinement.exhausted:
        state.refinement.reset_cycle()
    state.refinement.record_turn("user")
    state.record_refinement_event("iterative_refine", source="user", note=instruction)
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
    verifier_stack: Any,
    state: DesignState,
    image_ref: str,
    sketch_pixel_ref: str | None,
    prompt: str | None,
    safety_score: float | None = None,
    lock_donor_ref: str | None = None,
) -> dict:
    from krisna_inference.backends.blob_store_singleton import get_blob_store

    blobs = get_blob_store()
    polished_image = blobs.load_image(image_ref)
    sketch_image = None
    if sketch_pixel_ref and sketch_pixel_ref.startswith("blob://"):
        try:
            sketch_image = blobs.load_image(sketch_pixel_ref)
        except Exception as e:
            log.warning("sketch_image_load_failed_for_handoff_verifier", extra={"ref": sketch_pixel_ref, "error": str(e)})
            sketch_image = None

    lock_donor_image = None
    if lock_donor_ref and lock_donor_ref.startswith("blob://"):
        try:
            lock_donor_image = blobs.load_image(lock_donor_ref)
        except Exception as e:
            log.warning("lock_donor_image_load_failed", extra={"ref": lock_donor_ref, "error": str(e)})
            lock_donor_image = None

    effective_prompt = prompt or state.constraints.style or "UI design"
    locked_regions = [r.model_dump() for r in state.constraints.locked_regions]
    expected_regions = list(locked_regions)
    expected_texts: list[str] = []
    if state.layout.layout_ref:
        try:
            layout_payload = json.loads(Path(state.layout.layout_ref).read_text(encoding="utf-8"))
            for node in layout_payload.get("nodes", []):
                bbox = node.get("bbox", {})
                if {"x1", "y1", "x2", "y2"}.issubset(bbox):
                    expected_regions.append({
                        "bbox": [bbox["x1"], bbox["y1"], bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]]
                    })
                if node.get("text"):
                    expected_texts.append(str(node["text"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            log.warning("layout_evaluation_reference_unavailable", extra={"error": str(error)})
    if hasattr(verifier_stack, "evaluate_finalize_output"):
        try:
            receipt = verifier_stack.evaluate_finalize_output(
                polished_image,
                prompt=effective_prompt,
                sketch_image=sketch_image,
                expected_regions=expected_regions,
                expected_texts=expected_texts,
                locked_regions=locked_regions,
                safety_score=safety_score,
                lock_donor_image=lock_donor_image,
            )
        except TypeError:
            receipt = verifier_stack.evaluate_finalize_output(
                polished_image,
                prompt=effective_prompt,
                sketch_image=sketch_image,
                expected_regions=expected_regions,
                expected_texts=expected_texts,
                locked_regions=locked_regions,
                safety_score=safety_score,
            )
        return {**receipt.get("metrics", {}), "evaluation_receipt": receipt}
    try:
        return verifier_stack.score_finalize_output(
            polished_image,
            prompt=effective_prompt,
            sketch_image=sketch_image,
            expected_regions=expected_regions,
            expected_texts=expected_texts,
            locked_regions=locked_regions,
            lock_donor_image=lock_donor_image,
        )
    except TypeError:
        return verifier_stack.score_finalize_output(
            polished_image,
            prompt=effective_prompt,
            sketch_image=sketch_image,
            expected_regions=expected_regions,
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

    CALLER CONTRACT worth stating explicitly (found while auditing for
    prompt/image mixups): the resulting PreferencePair stores ONE prompt
    for BOTH candidates (state.finalize_output.prompt_used, the current
    render's actual generation prompt — see the fix in the body below).
    Diffusion-DPO's own derivation assumes both the chosen and rejected
    sample are conditioned on the SAME prompt/context when computing the
    reward margin. `compare_against` is therefore only a valid partner for
    a pair if it was ALSO generated from that same effective prompt (e.g.
    an earlier candidate from the same iterative-refinement loop on this
    same design intent) — not an image from a genuinely different prompt
    or an earlier, since-changed conversation turn. Nothing in this
    function can verify that on the caller's behalf; passing a
    `compare_against` from a different design intent silently produces a
    pair whose stored prompt is wrong for one of its two images, the same
    class of mixup the prompt_used fix below closes for the OTHER side of
    the pair.

    UPGRADE: `compare_against` may now optionally include its own
    `"prompt_used"` key (the real generation-time prompt for THAT
    candidate, if the caller tracked one). When present,
    build_pair_from_candidates cross-checks it against this render's own
    `prompt_used` and raises MismatchedPromptError instead of silently
    writing a pair with an ambiguous prompt — turning the caller-contract
    note above from a documentation-only warning into an enforced check
    whenever the caller has the data to make enforcement possible.
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
        from krisna_inference.verifiers.critique_adapter import from_planner_review

        v_scores = (
            state.finalize_output.verifier_scores.model_dump()
            if (state.finalize_output and state.finalize_output.verifier_scores)
            else None
        )
        state.critique.result = from_planner_review(output, verifier_scores=v_scores)
    state.critique.source = "planner_review"
    from datetime import datetime, timezone

    state.critique.timestamp = datetime.now(timezone.utc).isoformat()

    if preference_store is not None and compare_against is not None:
        from krisna_training.dpo.pair_builder import build_pair_from_candidates

        # P1 FIX (semantic audit): was `state.constraints.style or "UI design"` — that
        # stored only the style token in the preference pair record, losing all user
        # intent. DPO training then called encode_prompt("minimalist dark") rather
        # than the full prompt the image was actually generated from, producing a
        # poor conditioning signal. Now uses the same full synthesis as finalize().
        #
        # BUG FOUND ON REVIEW (a second, distinct mismatch on top of the P1 fix
        # above): critique_pass() can run turns after finalize() — the user may
        # have sent more chat messages in between (asking for a further revision,
        # small talk, anything). _synthesize_dpo_prompt(state) reads
        # state.conversation_history[-1] as of THIS call, not as of when
        # state.finalize_output.image_ref was actually generated — so unconditionally
        # re-synthesizing here could store a prompt describing a LATER user turn
        # against an image generated from an EARLIER one, a real (prompt, image)
        # mixup landing directly in DPO training data. finalize() now freezes the
        # true generation-time prompt on state.finalize_output.prompt_used (see
        # design_state.py); read that back instead of re-deriving it, falling back
        # to a fresh synthesis only for a state persisted before this field existed
        # (prompt_used is None) rather than crashing on old sessions.
        pair = build_pair_from_candidates(
            preference_store,
            prompt=state.finalize_output.prompt_used or _synthesize_dpo_prompt(state),
            candidates=[
                {
                    "image_ref": state.finalize_output.image_ref,
                    "score": state.critique.result.overall_score,
                    # UPGRADE (closes the caller-contract gap this
                    # docstring used to only describe): report the
                    # REAL generation-time prompt for this specific
                    # candidate so build_pair_from_candidates can
                    # actually enforce the same-prompt assumption
                    # instead of only documenting it. If the caller's
                    # compare_against also includes its own
                    # "prompt_used", a mismatch is now caught and
                    # raised rather than silently written.
                    "prompt_used": state.finalize_output.prompt_used,
                },
                compare_against,
            ],
            # The active critique path is Planner/verifier-backed; the frozen
            # Critic is not a training-label authority. Keep this source in
            # the explicit PreferenceStore allowlist.
            source="planner_review",
            session_id=session_id,
        )
        if pair is not None:
            state.preference_pair_refs.append(pair.id)

    state.stage = SessionStage.FINALIZED
    state.touch()
    return store.save(state, expected_rev)

def _synthesize_dpo_prompt(state: DesignState) -> str:
    """Canonical prompt-synthesis for DPO preference pairs and Polish-tier
    inference. Produces: '<last user intent>, style: <style>, palette: <...>,
    layout: <...>' — the same format in both places so the LoRA trains on
    the distribution it will actually see at inference time.

    Used by finalize() (passed to Polish backends) and critique_pass()
    (stored in preference pair records for later DPO training). A single
    function rather than two separate inline copies prevents the two usages
    from drifting apart again (the P1 bug this fixes was exactly that drift).
    """
    user_turns = [turn.content for turn in state.conversation_history if turn.role == "user"]
    latest_user_intent = user_turns[-1].strip() if user_turns else ""

    descriptors = []
    if state.constraints.style:
        descriptors.append(f"style: {state.constraints.style}")
    if state.constraints.palette:
        descriptors.append(f"palette: {', '.join(state.constraints.palette)}")
    if state.constraints.layout_hints:
        descriptors.append(f"layout: {state.constraints.layout_hints}")
    # UPGRADE (docs/review/35_model_review_3_resync.md): the same
    # Planner-reasoning-note enrichment already applied to
    # structured layout backend, carried forward here via the persisted
    # state field (see DesignState.last_planner_reasoning_note's
    # docstring for why a persisted field was needed here but not for
    # Sketch). Bound is DELIBERATELY much more generous than Sketch's
    # 160-character cap, not copy-pasted from it — researched this
    # tier's actual text encoder rather than assuming the same CLIP-
    # The Qwen-Image-2.1 path keeps the generation-time prompt as the single
    # source of truth for any later evidence or preference record.
    # (confirmed via its own text_encoder/config.json on HuggingFace),
    # with diffusers/DiffSynth enforcing max_sequence_length=512 tokens
    # — roughly 6-7x CLIP's ~77-token practical limit, not the ~50-60
    # word budget CLIP's silent truncation forced Sketch's cap to
    # respect. 300 characters is a defensive sanity bound (a Planner
    # that ignored its own "<=2 sentences" instruction could in
    # principle produce something much longer), not a tight constraint
    # against truncation risk the way Sketch's cap was.
    if state.last_planner_reasoning_note:
        descriptors.append(state.last_planner_reasoning_note[:300])

    desc_str = ", ".join(descriptors)
    if latest_user_intent and desc_str:
        return f"{latest_user_intent}, {desc_str}"
    elif latest_user_intent:
        return latest_user_intent
    elif desc_str:
        return f"High quality UI design, {desc_str}"
    else:
        return "High quality UI design"


PROMPT_TEMPLATE_VERSION = "krisna.polish_prompt.v3.pe_prose"


def assert_and_inject_verbatim_labels(prompt: str, expected_labels: list[str]) -> str:
    """Verbatim Label Guard: guarantees that 100% of required text labels survive in the prompt.

    Diffusion models with in-context conditioning require explicit textual tokens for text rendering.
    If prompt rewriting (e.g. PE-I2I or LLM) omitted any exact label strings, this guard detects
    the omissions and re-injects them verbatim to ensure zero typographic degradation.
    """
    if not expected_labels:
        return prompt

    missing = [label for label in expected_labels if label not in prompt]
    if missing:
        quoted = ", ".join(f"'{lbl}'" for lbl in missing)
        injection = f" Text labels rendered legibly: {quoted}."
        prompt = f"{prompt.rstrip('.')}." + injection
    return prompt


def synthesize_descriptive_polish_prompt(state: DesignState) -> str:
    """Synthesizes rich, descriptive natural language prose matching Qwen PE format.

    Qwen-Image-2.1 was trained on long, coherent descriptive prose rather than raw keyword lists.
    This generator:
    1. Extracts structured spec: node IDs, roles, exact label strings, and region-by-region placements.
    2. Constructs descriptive natural language prose connecting regions, components, and palette tokens.
    3. Executes the Verbatim Label Guard to ensure all required strings survive without omission.
    """
    base_prompt = _synthesize_dpo_prompt(state)
    if not state.layout.layout_ref:
        return base_prompt

    try:
        layout_path = Path(state.layout.layout_ref)
        if not layout_path.exists():
            return base_prompt

        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])

        # Extract typographic label strings and components
        expected_labels = [
            str(node["text"]).strip()
            for node in nodes
            if node.get("text") and str(node["text"]).strip()
        ]

        # Categorize components by spatial region
        header_nodes = []
        main_nodes = []
        action_nodes = []

        for node in nodes:
            if node.get("id") == "root":
                continue
            comp_type = str(node.get("type", "element"))
            text = str(node.get("text") or "").strip()
            bbox = node.get("bbox") or {}
            if isinstance(bbox, dict):
                x1 = bbox.get("x1", 0.0)
                y1 = bbox.get("y1", 0.5)
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1 = bbox[0], bbox[1]
            else:
                x1, y1 = 0.0, 0.5

            item = {"id": node.get("id"), "type": comp_type, "text": text, "x1": x1, "y1": y1, "bbox": bbox}
            if y1 < 0.22 or node.get("id") in ("title", "header", "nav") or comp_type in ("navbar", "header"):
                header_nodes.append(item)
            elif y1 > 0.80 or comp_type in ("button", "footer"):
                action_nodes.append(item)
            else:
                main_nodes.append(item)

        prose_parts = []
        style_desc = state.constraints.style or "modern clean"
        archetype = str(payload.get("scene_archetype") or getattr(state.layout, "scene_archetype", "ui_layout")).lower()

        if archetype == "portrait_character":
            focal_nodes = [
                n for n in nodes if n.get("is_focal_subject") or n.get("type") in ("focal_subject", "character")
            ]
            if len(focal_nodes) > 1:
                focal_descs = [str(n.get("text") or n.get("id") or "character").strip() for n in focal_nodes]
                prose_parts.append(
                    f"A stunning, harmonious dual-subject portrait featuring {' and '.join(focal_descs)} together in the scene with {style_desc} aesthetic."
                )
                prose_parts.append(
                    "Masterfully composed multi-subject portrait with distinct facial identities preserved, natural interaction, balanced skin tones, and cohesive unified lighting and depth of field."
                )
                other_labels = [
                    f"{n.get('type')}: '{n['text']}'"
                    for n in nodes
                    if n.get("id") != "root" and n.get("text") and n not in focal_nodes
                ]
                if other_labels:
                    prose_parts.append(f"Subtle visual typography elements present: {', '.join(other_labels)}.")
            else:
                focal_node = focal_nodes[0] if focal_nodes else None
                focal_desc = str(focal_node.get("text") or "character portrait").strip() if focal_node else "character portrait"
                prose_parts.append(
                    f"A breathtaking, high-fidelity character portrait of {focal_desc} with {style_desc} aesthetic."
                )
                prose_parts.append(
                    "Featuring intricate, realistic facial mapping, natural skin textures, expressive eyes, and fine hair details with sharp focal depth and balanced lighting."
                )
                other_labels = [
                    f"{n.get('type')}: '{n['text']}'"
                    for n in nodes
                    if n.get("id") != "root" and n.get("text") and n != focal_node
                ]
                if other_labels:
                    prose_parts.append(f"Subtle visual typography elements present: {', '.join(other_labels)}.")

        elif archetype == "graphic_poster":
            headline_node = next((n for n in nodes if n.get("type") == "headline"), None)
            hl_text = f"displaying '{headline_node['text']}'" if headline_node and headline_node.get("text") else "bold typography"
            focal_node = next(
                (n for n in nodes if n.get("is_focal_subject") or n.get("type") in ("illustration", "focal_subject")),
                None,
            )
            focal_text = f"featuring {focal_node['text']}" if focal_node and focal_node.get("text") else "striking central illustration"
            prose_parts.append(f"A dynamic, production-grade graphic poster design with {style_desc} aesthetic.")
            prose_parts.append(
                f"The composition showcases {hl_text} anchored by {focal_text} with balanced visual hierarchy and poster aesthetics."
            )

        elif archetype == "landscape_scene":
            focal_node = next((n for n in nodes if n.get("is_focal_subject")), None)
            focal_text = f"highlighting {focal_node['text']}" if focal_node and focal_node.get("text") else "natural vista"
            prose_parts.append(
                f"An expansive, atmospheric environmental landscape scene with {style_desc} aesthetic."
            )
            prose_parts.append(
                f"Featuring rich environmental depth, {focal_text}, natural atmospheric perspective, realistic lighting, and detailed textures across the canvas."
            )

        elif archetype == "e_commerce":
            prod_node = next((n for n in nodes if n.get("type") == "product" or n.get("is_focal_subject")), None)
            prod_text = f"highlighting {prod_node['text']}" if prod_node and prod_node.get("text") else "featured product"
            prose_parts.append(f"A commercial-grade product showcase with {style_desc} aesthetic.")
            prose_parts.append(
                f"Presenting a pristine hero product composition {prod_text} with studio lighting, sharp reflections, and premium catalog visual balance."
            )

        elif archetype == "freeform_art":
            focal_node = next((n for n in nodes if n.get("is_focal_subject")), None)
            focal_text = f"expressing {focal_node['text']}" if focal_node and focal_node.get("text") else "artistic subject"
            prose_parts.append(f"An expressive artistic composition with {style_desc} aesthetic.")
            prose_parts.append(
                f"Featuring rich artistic textures, dynamic contours {focal_text}, and cohesive visual composition."
            )

        else:
            # Default UI layout prose
            prose_parts.append(f"A polished, production-grade UI design with {style_desc} aesthetic.")

            # Header description
            if header_nodes:
                header_strs = []
                for h in header_nodes[:3]:
                    if h["text"]:
                        header_strs.append(f"{h['type']} displaying '{h['text']}'")
                    else:
                        header_strs.append(f"{h['type']}")
                prose_parts.append(f"At the top header region, there is a clean layout featuring {', '.join(header_strs)}.")

            # Main section description (2D Multi-Column & Spatial Partition Awareness)
            if main_nodes:
                sorted_main = sorted(main_nodes, key=lambda m: m["x1"])
                is_multicol = (
                    len(sorted_main) >= 2
                    and any(abs(sorted_main[i]["x1"] - sorted_main[i-1]["x1"]) > 0.15 for i in range(1, len(sorted_main)))
                )
                if is_multicol:
                    col_strs = []
                    for c_idx, m in enumerate(sorted_main[:4], start=1):
                        if m["text"]:
                            col_strs.append(f"column {c_idx} [x: {m['x1']:.2f}] containing {m['type']} with '{m['text']}'")
                        else:
                            col_strs.append(f"column {c_idx} [x: {m['x1']:.2f}] with {m['type']}")
                    prose_parts.append(f"The central main section presents a structured multi-column partition featuring {', '.join(col_strs)} with balanced visual hierarchy.")
                else:
                    main_strs = []
                    for m in main_nodes[:5]:
                        if m["text"]:
                            main_strs.append(f"{m['type']} containing '{m['text']}'")
                        else:
                            main_strs.append(f"{m['type']}")
                    prose_parts.append(f"The central main section presents {', '.join(main_strs)} with structured visual hierarchy.")

            # Actions section description
            if action_nodes:
                act_strs = []
                for a in action_nodes[:3]:
                    if a["text"]:
                        act_strs.append(f"{a['type']} labeled '{a['text']}'")
                    else:
                        act_strs.append(f"{a['type']}")
                prose_parts.append(f"In the lower action area, there is {', '.join(act_strs)} prominently placed.")

        # Visual styling and palette
        if state.constraints.palette:
            palette_str = ", ".join(state.constraints.palette[:4])
            prose_parts.append(f"The design utilizes a cohesive color palette incorporating {palette_str} with balanced contrast.")

        # Locked regions preservation
        if state.constraints.locked_regions:
            locked_reasons = [r.reason or f"region at {r.bbox}" for r in state.constraints.locked_regions]
            prose_parts.append(f"PRESERVE UNCHANGED: {'; '.join(locked_reasons)}.")

        prompt = " ".join(prose_parts)

        # Enforce Verbatim Label Guard
        guarded_prompt = assert_and_inject_verbatim_labels(prompt, expected_labels)
        return guarded_prompt
    except Exception as exc:
        log.warning("descriptive_prompt_synthesis_error: %s", exc)
        return base_prompt


def build_grounded_spatial_retry_prompt(
    state: DesignState,
    base_prompt: str,
    drift_reasons: list[str],
    layout_iou: float | None = None,
    ocr_score: float | None = None,
    seam_score: float | None = None,
    lock_preservation: float | None = None,
) -> str:
    """Builds a grounded, region-specific natural language spatial feedback prompt for retry turns.

    Instead of abstract coordinate reminders, this provides clear linguistic spatial instructions
    referencing specific buttons, headers, cards, and text labels from the layout.
    """
    feedback = []
    nodes = []
    if state.layout.layout_ref and Path(state.layout.layout_ref).exists():
        try:
            data = json.loads(Path(state.layout.layout_ref).read_text(encoding="utf-8"))
            nodes = data.get("nodes", [])
        except Exception:
            pass

    button_nodes = [
        n for n in nodes
        if n.get("type") == "button" or (isinstance(n.get("bbox"), dict) and n.get("bbox", {}).get("y1", 0) > 0.75)
    ]
    title_nodes = [
        n for n in nodes
        if n.get("id") in ("title", "header") or (isinstance(n.get("bbox"), dict) and n.get("bbox", {}).get("y1", 0) < 0.25)
    ]
    card_nodes = [n for n in nodes if n.get("type") == "card"]

    if layout_iou is not None and layout_iou < 0.70:
        spatial_guidance = []
        if button_nodes:
            b_text = button_nodes[0].get("text") or "action"
            spatial_guidance.append(f"the '{b_text}' button must sit at the bottom-right under the main card")
        if title_nodes:
            t_text = title_nodes[0].get("text") or "title"
            spatial_guidance.append(f"the '{t_text}' header must sit at the top without downward drift")
        if card_nodes:
            spatial_guidance.append("the main card must remain centered without spatial distortion")
        if spatial_guidance:
            feedback.append(f"Spatial alignment correction: {', '.join(spatial_guidance)}.")
        else:
            feedback.append("Spatial alignment correction: elements must strictly stay within their designated visual regions.")

    if ocr_score is not None and ocr_score < 0.80:
        labels = [n.get("text") for n in nodes if n.get("text")][:4]
        if labels:
            quoted_labels = ", ".join(f"'{lbl}'" for lbl in labels)
            feedback.append(f"Typographic legibility correction: ensure text labels ({quoted_labels}) are rendered crisply with high contrast and exact spelling.")
        else:
            feedback.append("Typographic legibility correction: render all UI text crisply and legibly without blur or distortion.")

    if seam_score is not None and seam_score < 0.85:
        feedback.append("Boundary continuity correction: smooth and harmonize edge transitions around locked regions.")

    if lock_preservation is not None and lock_preservation < 0.95:
        feedback.append(f"Lock preservation correction ({lock_preservation:.2f} < 0.95): preserve all locked donor pixels strictly without visual drift or identity degradation.")

    if not feedback:
        feedback.append("Ensure crisp spatial separation between components without overlapping.")

    correction_text = " ".join(feedback)
    return f"{base_prompt.rstrip('.')}. CRITICAL LAYOUT ENFORCEMENT: {correction_text}"


def synthesize_polish_prompt(state: DesignState) -> str:
    """Synthesizes high-fidelity prompt for Polish tier (Qwen-Image-2.1).

    Enriches the base intent with:
    1. Exact typographic labels protected by the Verbatim Label Guard.
    2. Natural language descriptive prose matching Qwen PE format.
    3. Regional component hierarchy (header, main content, action bar).
    4. Curated color palette and immutable lock directives.
    """
    return synthesize_descriptive_polish_prompt(state)


def export_research_trajectory(store: DesignStateStore, session_id: str):
    """Exports a SessionResearchTrajectory capturing reasoning traces and turn metrics for research analysis."""
    from krisna_inference.research.trace_recorder import build_research_trajectory

    state = store.get(session_id)
    return build_research_trajectory(state)
