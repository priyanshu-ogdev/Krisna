"""Phase 5 Comprehensive Verification Test Suite:
FlowEdit-Style Iterative Refinement, Dual-Layer Mask Conditioning, & VLA Design Action Grammar.

Research basis:
- FlowEdit (Kulikov et al., ICCV 2025): 8-step flow-matching ODE inversion & editing.
- MasaCtrl (Cao et al., ICCV 2023): Mutual self-attention consistency for non-rigid edits.
- ShowUI (Lin et al., CVPR 2025): Action-token grammar for GUI elements.
- SeeClick (Cheng et al., ACL 2024): Visual-GUI grounding and precision manipulation.
- InstructLayout (Wen et al., IEEE TPAMI 2025): Instruction-driven UI layout refinement.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from krivex_inference.backends.planner_schema import (
    ActionType,
    AddComponentAction,
    DesignAction,
    DesignStateDelta,
    LockRegionAction,
    MoveComponentAction,
    PlannerResponse,
    RemoveComponentAction,
    ReorderHierarchyAction,
    ResizeAction,
    SetLayoutDirectionAction,
    SetPaletteAction,
    SetStyleAction,
    SetTextAction,
)
from krivex_inference.backends.polish_backend_qwen import FlowODEIntegrator
from krivex_inference.orchestrator import flows
from krivex_inference.orchestrator.constraint_merge import apply_constraint_updates
from krivex_inference.orchestrator.design_state import DesignState, SessionStage
from krivex_inference.orchestrator.store import DesignStateStore
from krivex_inference.orchestrator.swap_orchestrator import SwapOrchestrator
from krivex_inference.verifiers.lock_preservation import LockPreservationVerifier


# ----------------------------------------------------------------------
# 1. VLA Design Action Grammar & Schema Tests
# ----------------------------------------------------------------------


def test_vla_action_models_and_schema_validation():
    """Verify typed DesignAction models and PlannerResponse serialization."""
    actions = [
        SetStyleAction(style="glassmorphism"),
        SetPaletteAction(palette=["#0f172a", "#38bdf8"]),
        LockRegionAction(bbox=[0.0, 0.0, 1.0, 0.15], reason="fixed navigation header"),
        ResizeAction(target_id="hero_banner", bbox=[0.0, 0.15, 1.0, 0.45]),
        AddComponentAction(component_type="button", bbox=[0.4, 0.5, 0.6, 0.58], props={"label": "Checkout"}),
        RemoveComponentAction(target_id="legacy_footer_ad"),
    ]

    delta = DesignStateDelta(
        stage="refining",
        constraint_updates={"layout_hints": "compact density"},
        actions=actions,
        reasoning_note="Phase 5 VLA action-grammar refinement.",
    )
    resp = PlannerResponse(
        reply_text="Refinement plan constructed.",
        design_state_delta=delta,
    )

    dumped = resp.model_dump()
    assert dumped["design_state_delta"]["stage"] == "refining"
    assert len(dumped["design_state_delta"]["actions"]) == 6
    assert dumped["design_state_delta"]["actions"][0]["action_type"] == "SET_STYLE"
    assert dumped["design_state_delta"]["actions"][0]["style"] == "glassmorphism"
    assert dumped["design_state_delta"]["actions"][2]["action_type"] == "LOCK_REGION"
    assert dumped["design_state_delta"]["actions"][2]["reason"] == "fixed navigation header"

    # Round-trip validate from dict payload
    reloaded = PlannerResponse.model_validate(dumped)
    assert len(reloaded.design_state_delta.actions) == 6
    assert reloaded.design_state_delta.stage == "refining"


def test_apply_constraint_updates_with_action_list():
    """Verify apply_constraint_updates correctly merges heterogeneous actions."""
    base_constraints = {
        "style": "clean minimal",
        "palette": ["#ffffff", "#000000"],
        "layout_hints": "desktop wide",
        "locked_regions": [],
    }

    actions = [
        SetStyleAction(style="dark brutalist"),
        SetPaletteAction(palette=["#121212", "#00ff66"]),
        LockRegionAction(bbox=[0.05, 0.05, 0.25, 0.15], reason="brand logo"),
        ResizeAction(target_id="sidebar", bbox=[0.0, 0.0, 0.2, 1.0]),
        AddComponentAction(component_type="search_bar", bbox=[0.3, 0.05, 0.7, 0.12]),
        RemoveComponentAction(target_id="cookie_banner"),
    ]

    merged = apply_constraint_updates(base_constraints, actions)

    assert merged["style"] == "dark brutalist"
    assert merged["palette"] == ["#121212", "#00ff66"]
    assert len(merged["locked_regions"]) == 1
    assert merged["locked_regions"][0]["bbox"] == [0.05, 0.05, 0.25, 0.15]
    assert merged["locked_regions"][0]["reason"] == "brand logo"
    assert "Resize sidebar to [0.0, 0.0, 0.2, 1.0]" in merged["layout_hints"]
    assert "Add search_bar at [0.3, 0.05, 0.7, 0.12]" in merged["layout_hints"]
    assert "Remove cookie_banner" in merged["layout_hints"]


def test_apply_constraint_updates_with_nested_actions_in_dict():
    """Verify constraint_updates dict containing 'actions' list is applied cleanly."""
    base_constraints = {
        "style": "light",
        "palette": ["#ffffff"],
        "layout_hints": "",
        "locked_regions": [],
    }

    update_payload = {
        "style": "flat",
        "actions": [
            {"action_type": "SET_STYLE", "style": "cyberpunk 2077"},
            {"action_type": "LOCK_REGION", "bbox": [0.0, 0.8, 1.0, 0.2], "reason": "footer bar"},
        ],
    }

    merged = apply_constraint_updates(base_constraints, update_payload)
    assert merged["style"] == "cyberpunk 2077"
    assert len(merged["locked_regions"]) == 1
    assert merged["locked_regions"][0]["reason"] == "footer bar"


# ----------------------------------------------------------------------
# 2. FlowEdit Iterative Refinement & Prompt Runaway Prevention
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iterative_refine_happy_path(
    started_orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    finalized_state: DesignState,
):
    """Test standard 8-step FlowEdit iterative refinement turn."""
    sid = finalized_state.session_id
    prior_render = finalized_state.finalize_output.image_ref

    from krivex_inference.orchestrator.design_state import CritiqueResult
    # Populate a dummy critique to verify it gets invalidated
    finalized_state.critique.requested = True
    finalized_state.critique.result = CritiqueResult(
        critique_source="planner_review",
        overall_score=0.82,
    )
    finalized_state.touch()
    store.save(finalized_state, expected_revision=finalized_state.revision - 1)

    refined_state = await flows.iterative_refine(
        started_orchestrator,
        store,
        sid,
        instruction="increase contrast of the navigation bar and make buttons rounded",
        num_inference_steps=8,
    )

    assert refined_state.stage == SessionStage.FINALIZED
    assert refined_state.finalize_output.image_ref is not None
    assert "Refinement edit: increase contrast of the navigation bar" in refined_state.finalize_output.prompt_used

    # Critique invalidated
    assert refined_state.critique.requested is False
    assert refined_state.critique.result is None

    # Refinement event recorded
    assert len(refined_state.refinement.events) >= 1
    last_event = refined_state.refinement.events[-1]
    assert last_event.event_type == "iterative_refine"
    assert "increase contrast" in last_event.note


@pytest.mark.asyncio
async def test_iterative_refine_prevents_prompt_runaway(
    started_orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    finalized_state: DesignState,
):
    """Verify repeated refinement calls do not concatenate runaway prefixes."""
    sid = finalized_state.session_id

    # Refinement 1
    st1 = await flows.iterative_refine(
        started_orchestrator, store, sid, instruction="make CTA green"
    )
    prompt1 = st1.finalize_output.prompt_used
    assert prompt1.count("Refinement edit:") == 1

    # Refinement 2
    st2 = await flows.iterative_refine(
        started_orchestrator, store, sid, instruction="make secondary buttons outline"
    )
    prompt2 = st2.finalize_output.prompt_used
    assert prompt2.count("Refinement edit:") == 1
    assert "make secondary buttons outline" in prompt2
    assert "make CTA green" not in prompt2


@pytest.mark.asyncio
async def test_iterative_refine_ineligible_raises_flow_error(
    started_orchestrator: SwapOrchestrator,
    store: DesignStateStore,
):
    """Verify attempting to refine a session without an existing render raises FlowError."""
    fresh_session = store.create()
    with pytest.raises(flows.FlowError) as excinfo:
        await flows.iterative_refine(
            started_orchestrator,
            store,
            fresh_session.session_id,
            instruction="make header sticky",
        )
    assert "not refine-eligible" in str(excinfo.value)


@pytest.mark.asyncio
async def test_iterative_refine_safety_gate_blocking(
    started_orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    finalized_state: DesignState,
    monkeypatch,
    tmp_path,
):
    """Verify that failing the safety gate blocks the refinement and cleans up the blob."""
    from krivex_inference.orchestrator.model_registry import Tier
    import krivex_inference.backends.blob_store_singleton as blob_singleton
    monkeypatch.setenv("KRIVEX_BLOB_ROOT", str(tmp_path / "blobs"))
    monkeypatch.setattr(blob_singleton, "_instance", None)

    # Create real blob image so verifier stack safety check is executed
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    blob_ref = blob_singleton.get_blob_store().save_image(img, prefix="polish_test")

    finalized_state.finalize_output.image_ref = blob_ref
    finalized_state.touch()
    store.save(finalized_state, expected_revision=finalized_state.revision - 1)

    async def fake_polish_run(**kwargs):
        return {"tier": "polish_default", "image_ref": blob_ref, "verifier_scores": {}}

    started_orchestrator.backends[Tier.POLISH_DEFAULT].run = fake_polish_run

    class MockFailingVerifierStack:
        def safety_gate(self, image, min_safety_score=0.9):
            return False, 0.42

    with pytest.raises(flows.FlowError, match="safety gate failed"):
        await flows.iterative_refine(
            started_orchestrator,
            store,
            finalized_state.session_id,
            instruction="make header red",
            verifier_stack=MockFailingVerifierStack(),
            min_safety_score=0.9,
        )
    # Session state restored to finalized
    curr = store.get(finalized_state.session_id)
    assert curr.stage == SessionStage.FINALIZED


# ----------------------------------------------------------------------
# 3. Dual-Layer Mask Conditioning & 0.00% Drift Lock Compositing
# ----------------------------------------------------------------------


def test_dual_layer_mask_conditioning_and_compositor():
    """Verify soft inpaint mask and bitwise invariance (0.00% pixel drift) on locked regions."""
    from krivex_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krivex_inference.orchestrator.model_registry import ModelSpec, Tier

    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=16.0, quantization="bf16")
    backend = QwenImage21Backend(spec)

    w, h = 200, 200
    source_img = Image.new("RGB", (w, h), color=(50, 100, 150))
    # Draw some texture in the locked region
    from PIL import ImageDraw
    draw = ImageDraw.Draw(source_img)
    draw.rectangle([20, 20, 80, 80], fill=(220, 20, 60))

    locked_regions = [
        {"bbox": [20, 20, 60, 60], "reason": "primary logo"}
    ]

    # 1. Soft inpainting mask check (255 = edit, 0 = locked)
    inpaint_mask = backend._build_inpaint_mask((w, h), locked_regions)
    mask_arr = np.array(inpaint_mask)
    # Locked region [20:80, 20:80] must be 0
    assert np.all(mask_arr[20:80, 20:80] == 0)
    # Unlocked region (e.g. outside) must be 255
    assert mask_arr[0, 0] == 255
    assert mask_arr[150, 150] == 255

    # 2. Rendered image with drastically modified background
    rendered_img = Image.new("RGB", (w, h), color=(255, 255, 255))

    # Apply lock compositor
    composited = backend._apply_locked_regions(source_img, rendered_img, locked_regions)

    # 3. Exact bitwise invariance check on locked region
    src_crop = np.array(source_img.crop((20, 20, 80, 80)))
    comp_crop = np.array(composited.crop((20, 20, 80, 80)))
    mse = np.mean((src_crop.astype(float) - comp_crop.astype(float)) ** 2)
    max_drift = np.max(np.abs(src_crop.astype(int) - comp_crop.astype(int)))
    assert mse == 0.0, f"Expected 0.00% drift, got MSE={mse}"
    assert max_drift == 0, f"Expected exact pixel preservation, got max_drift={max_drift}"

    # 4. LockPreservationVerifier verification
    score = LockPreservationVerifier().score(source_img, composited, locked_regions)
    assert score == 1.0


@pytest.mark.asyncio
async def test_iterative_refine_with_actions_end_to_end(
    started_orchestrator: SwapOrchestrator,
    store: DesignStateStore,
    finalized_state: DesignState,
):
    """Verify iterative_refine with explicit DesignAction objects mutates constraints and preserves locks."""
    sid = finalized_state.session_id

    actions = [
        SetStyleAction(style="nordic minimal"),
        LockRegionAction(bbox=[0.0, 0.0, 1.0, 0.1], reason="top nav bar"),
    ]

    refined_state = await flows.iterative_refine(
        started_orchestrator,
        store,
        sid,
        instruction="change hero text to bold and tighten spacing",
        actions=actions,
    )

    assert refined_state.constraints.style == "nordic minimal"
    assert len(refined_state.constraints.locked_regions) >= 1
    assert refined_state.constraints.locked_regions[-1].reason == "top nav bar"


def test_flowedit_ode_integrator_operator():
    """Verify FlowEdit ODE step operator (Kulikov et al., ICCV 2025, Eq. 4)."""
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    v_edit = np.array([0.5, 0.8, -0.2], dtype=float)
    v_src = np.array([0.1, 0.3, 0.4], dtype=float)
    dt = 0.125  # 1/8 step

    # x_{t+dt} = x + dt * (v_edit - v_src)
    expected = x + dt * (v_edit - v_src)
    result = FlowODEIntegrator.flowedit_step(x, v_edit, v_src, dt)
    np.testing.assert_allclose(result, expected)

    # Identical vector fields result in exactly zero drift
    zero_drift = FlowODEIntegrator.flowedit_step(x, v_src, v_src, dt)
    np.testing.assert_allclose(zero_drift, x)


def test_extended_vla_action_models_and_merge():
    """Verify extended VLA actions: MoveComponentAction, SetTextAction, SetLayoutDirectionAction, ReorderHierarchyAction."""
    base_constraints = {
        "style": "clean",
        "palette": ["#ffffff"],
        "layout_hints": "",
        "locked_regions": [],
    }

    actions = [
        MoveComponentAction(target_id="cta_button", bbox=[0.3, 0.4, 0.4, 0.1]),
        SetTextAction(target_id="headline", text_content="Next Gen Generative UI"),
        SetLayoutDirectionAction(target_id="card_grid", layout_direction="grid"),
        ReorderHierarchyAction(target_id="modal_dialog", hierarchy_position="front"),
    ]

    merged = apply_constraint_updates(base_constraints, actions)
    assert "Move cta_button to [0.3, 0.4, 0.4, 0.1]" in merged["layout_hints"]
    assert "Set text of headline to 'Next Gen Generative UI'" in merged["layout_hints"]
    assert "Set card_grid direction to grid" in merged["layout_hints"]
    assert "Reorder modal_dialog to front" in merged["layout_hints"]


def test_bounding_box_clamping_and_bounds_invariance():
    """Verify out-of-bounds bounding boxes are clamped gracefully to prevent rendering crashes."""
    base_constraints = {
        "style": "clean",
        "palette": [],
        "layout_hints": "",
        "locked_regions": [],
    }

    # Bounding box exceeding 1.0 (e.g. x=0.2, w=0.9 -> x+w=1.1)
    actions = [
        LockRegionAction(bbox=[0.2, 0.3, 0.9, 0.8], reason="overflow header"),
    ]

    merged = apply_constraint_updates(base_constraints, actions)
    locked = merged["locked_regions"][0]
    bx, by, bw, bh = locked["bbox"]
    assert bx + bw <= 1.0001
    assert by + bh <= 1.0001
    assert bw == 0.8  # clamped from 0.9 to 1.0 - 0.2 = 0.8
    assert bh == 0.7  # clamped from 0.8 to 1.0 - 0.3 = 0.7

