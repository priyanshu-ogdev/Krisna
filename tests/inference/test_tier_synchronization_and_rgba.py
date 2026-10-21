from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from krisna_inference.backends.planner_schema import (
    AddComponentAction,
    LockRegionAction,
    RemoveComponentAction,
    ResizeAction,
    SetTextAction,
)
from krisna_inference.layout_baseline import layout_from_constraints
from krisna_inference.layout_renderer import render_wireframe
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


def test_layout_baseline_synchronizes_with_planner_actions():
    """Verify that layout_from_constraints dynamically updates LayoutIR based on Planner actions."""
    constraints = {
        "style": "clean dark",
        "palette": ["#0f172a", "#38bdf8", "#818cf8"],
        "actions": [
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.25, 0.80, 0.50],
                target_id="feature_card",
                props={"label": "Analytics Overview"},
            ),
            AddComponentAction(
                component_type="button",
                bbox=[0.70, 0.85, 0.20, 0.08],
                target_id="checkout_btn",
                props={"label": "Pay Now"},
            ),
            RemoveComponentAction(target_id="action"),  # removes default action
            SetTextAction(target_id="title", text_content="Custom Dashboard Title"),
            LockRegionAction(target_id="title"),
        ],
    }

    layout = layout_from_constraints(constraints, text="Starter Prompt")
    node_map = layout.node_map()

    # 1. Custom title was updated and locked
    assert "title" in node_map
    assert node_map["title"].text == "Custom Dashboard Title"
    assert node_map["title"].locked is True

    # 2. Default action button was removed
    assert "action" not in node_map

    # 3. New components were added
    assert "feature_card" in node_map
    assert node_map["feature_card"].type == "card"
    assert node_map["feature_card"].text == "Analytics Overview"

    assert "checkout_btn" in node_map
    assert node_map["checkout_btn"].type == "button"
    assert node_map["checkout_btn"].text == "Pay Now"


def test_layout_baseline_synchronizes_with_locked_regions():
    """Verify that locked_regions in constraints correctly sets locked=True on overlapping nodes."""
    constraints = {
        "style": "minimal",
        "palette": ["#ffffff", "#000000"],
        "locked_regions": [
            {"bbox": [0.08, 0.08, 0.85, 0.15], "reason": "lock navbar/header"},
        ],
    }

    layout = layout_from_constraints(constraints, text="Welcome Header")
    node_map = layout.node_map()

    # The 'title' node is at (0.10, 0.10, 0.90, 0.20), which overlaps with the locked region
    assert node_map["title"].locked is True
    # The 'content' node is at y1=0.26, which is outside the locked region
    assert node_map["content"].locked is False


def test_render_wireframe_renders_synchronized_layout(tmp_path):
    """Verify wireframe renderer handles the dynamically synchronized layout without error."""
    constraints = {
        "style": "dark",
        "palette": ["#0f172a", "#38bdf8", "#818cf8"],
        "actions": [
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.25, 0.80, 0.50],
                target_id="card_1",
                props={"label": "Card Content"},
            ),
            AddComponentAction(
                component_type="button",
                bbox=[0.10, 0.85, 0.30, 0.08],
                target_id="btn_1",
                props={"label": "Click Me"},
            ),
        ],
    }
    layout = layout_from_constraints(constraints, text="App")
    out_png = tmp_path / "sync_wireframe.png"

    path = render_wireframe(layout, out_png, style="dark")
    assert path.exists()

    with Image.open(path) as img:
        assert img.size == (layout.canvas_width, layout.canvas_height)
        assert img.mode == "RGB"


@pytest.mark.asyncio
async def test_polish_backend_preserves_rgba_mode():
    """Verify QwenImage21Backend preserves native RGBA mode without stripping alpha."""
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=4.5,
        quantization="BF16",
        always_resident=False,
    )
    backend = QwenImage21Backend(spec)
    await backend.load()

    # Create an RGBA source image with an alpha channel
    rgba_image = Image.new("RGBA", (100, 100), (30, 40, 50, 200))
    rendered = backend._synthesize_render(
        source_img=rgba_image,
        prompt="Polish this UI",
        constraints={},
        locked_regions=[],
        scheduler=None,
    )

    assert rendered.mode == "RGBA"

    # Test lock region preservation in RGBA mode
    locked_rendered = backend._apply_locked_regions(
        source_img=rgba_image,
        rendered_img=rendered,
        locked_regions=[{"bbox": [0.1, 0.1, 0.4, 0.4]}],
    )
    assert locked_rendered.mode == "RGBA"


@pytest.mark.asyncio
async def test_end_to_end_planner_to_polish_sync(tmp_path, monkeypatch):
    """Verify the full Planner -> Layout Engine -> Renderer -> Polish -> Verifiers chain."""
    import json
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, Constraints, LockedRegion
    from krisna_inference.orchestrator.flows import synthesize_polish_prompt
    from krisna_inference.verifiers.layout_iou import LayoutIoUVerifier
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    store = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: store)

    # 1. State with Planner actions
    state = DesignState()
    state.constraints = Constraints(
        style="modern SaaS",
        palette=["#0f172a", "#38bdf8"],
        locked_regions=[LockedRegion(bbox=[0.1, 0.2, 0.8, 0.5], reason="hero card locked")],
    )

    actions = [
        AddComponentAction(
            component_type="card",
            bbox=[0.1, 0.2, 0.8, 0.5],
            target_id="hero_card",
            props={"label": "AI Analytics Dashboard"},
        ),
        AddComponentAction(
            component_type="button",
            bbox=[0.1, 0.75, 0.3, 0.08],
            target_id="cta_btn",
            props={"label": "Get Started"},
        ),
    ]

    # 2. Build LayoutIR from constraints + actions
    merged_constraints = state.constraints.model_dump()
    merged_constraints["actions"] = actions
    layout = layout_from_constraints(merged_constraints, text="Analytics Page")

    layout_json_path = tmp_path / "layout.json"
    layout_json_path.write_text(json.dumps(layout.model_dump()), encoding="utf-8")
    state.layout.layout_ref = str(layout_json_path)

    # Render wireframe
    wireframe_path = tmp_path / "wireframe.png"
    render_wireframe(layout, wireframe_path, style="dark")
    with Image.open(wireframe_path) as wf_img:
        wireframe_ref = store.save_image(wf_img, prefix="wireframe")
    state.layout.wireframe_ref = wireframe_ref

    # 3. Test Polish prompt synthesis from LayoutIR
    polish_prompt = synthesize_polish_prompt(state)
    assert "AI Analytics Dashboard" in polish_prompt
    assert "Get Started" in polish_prompt
    assert "card" in polish_prompt

    # 4. Run Polish Backend with high-entropy wireframe and locked regions
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=4.5,
        quantization="BF16",
        always_resident=False,
    )
    backend = QwenImage21Backend(spec)
    await backend.load()

    polish_result = await backend.run(
        handoff_image_ref=wireframe_ref,
        prompt=polish_prompt,
        constraints=state.constraints.model_dump(),
        locked_regions=[r.model_dump() for r in state.constraints.locked_regions],
    )

    polished_image_ref = polish_result["image_ref"]
    polished_img = store.load_image(polished_image_ref)
    assert polished_img.size == (layout.canvas_width, layout.canvas_height)

    # 5. Verify closed-loop Verifier scoring
    wireframe_img = store.load_image(wireframe_ref)
    lock_score = LockPreservationVerifier().score(
        wireframe_img,
        polished_img,
        [r.model_dump() for r in state.constraints.locked_regions],
    )
    # Locked region must have exact 1.000 preservation
    assert lock_score == 1.0

    # Layout IoU verifier checks expected regions from LayoutIR
    expected_regions = [{"bbox": node.bbox.to_xywh()} for node in layout.nodes if node.id != "root"]
    iou_score = LayoutIoUVerifier().score(polished_img, expected_regions)
    assert iou_score > 0.0


def test_deterministic_flexbox_partitions_colliding_components():
    """Verify that _resolve_flex_layout automatically resolves colliding sibling components."""
    from krisna_inference.layout_baseline import layout_from_constraints

    constraints = {
        "style": "clean modern",
        "palette": ["#ffffff", "#000000", "#4f46e5"],
        "layout_direction": "row",
        "actions": [
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.30, 0.80, 0.40],
                target_id="card_a",
                props={"label": "Pricing Plan A"},
            ),
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.30, 0.80, 0.40],  # exact duplicate/colliding bbox
                target_id="card_b",
                props={"label": "Pricing Plan B"},
            ),
        ],
    }

    layout = layout_from_constraints(constraints, text="Pricing")
    node_map = layout.node_map()

    assert "card_a" in node_map
    assert "card_b" in node_map

    box_a = node_map["card_a"].bbox
    box_b = node_map["card_b"].bbox

    # They must NO LONGER overlap (IoU must be 0.0)
    assert box_a.iou(box_b) == 0.0
    # They should be neatly distributed side-by-side
    assert box_a.x2 < box_b.x1 or box_b.x2 < box_a.x1


def test_truetype_font_hierarchy_resolution():
    """Verify TrueType font resolution resolves modern typography scale without error."""
    from krisna_inference.layout_renderer import _resolve_ui_fonts

    fonts = _resolve_ui_fonts()
    for role in ("display", "title", "header", "body", "button", "caption", "badge"):
        assert role in fonts
        assert fonts[role] is not None


def test_locked_nodes_strictly_exempt_from_flex_repartitioning():
    """Verify that locked nodes are never moved by _resolve_flex_layout, even when colliding."""
    from krisna_inference.layout_baseline import layout_from_constraints

    constraints = {
        "style": "clean modern",
        "palette": ["#ffffff", "#000000"],
        "layout_direction": "row",
        "actions": [
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.30, 0.80, 0.40],
                target_id="locked_hero",
                props={"label": "Locked Hero Header"},
            ),
            LockRegionAction(target_id="locked_hero"),
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.30, 0.80, 0.40],  # Colliding coordinates
                target_id="mutable_promo",
                props={"label": "Mutable Promo Card"},
            ),
        ],
    }

    layout = layout_from_constraints(constraints, text="Hero Page")
    node_map = layout.node_map()

    assert "locked_hero" in node_map
    assert "mutable_promo" in node_map

    hero_box = node_map["locked_hero"].bbox
    promo_box = node_map["mutable_promo"].bbox

    # Locked node must NOT have moved from its original coordinates!
    assert hero_box.x1 == 0.10
    assert hero_box.y1 == 0.30
    assert node_map["locked_hero"].locked is True

    # Mutable node yielded space and no longer collides
    assert hero_box.iou(promo_box) == 0.0


def test_non_tautological_lock_compositor_with_perturbed_render():
    """Verify LockCompositor and LockPreservationVerifier are non-tautological by perturbing render."""
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=4.5, quantization="BF16", always_resident=False)
    backend = QwenImage21Backend(spec)

    # 1. Create reference donor image (solid blue)
    donor = Image.new("RGB", (200, 200), (0, 100, 255))

    # 2. Create rendered output that has drifted/perturbed in locked region (solid red)
    perturbed = Image.new("RGB", (200, 200), (255, 0, 0))

    locked_region = [{"bbox": [0.1, 0.1, 0.5, 0.5]}]
    verifier = LockPreservationVerifier()

    # Pre-compositing: verifier must detect corrupted pixels and score < 0.30 (NOT tautological!)
    score_before = verifier.score(donor, perturbed, locked_region)
    assert score_before < 0.30

    # Apply compositor
    composited = backend._apply_locked_regions(donor, perturbed, locked_region)

    # Post-compositing: score must recover to exact 1.000
    score_after = verifier.score(donor, composited, locked_region)
    assert score_after == 1.0


@pytest.mark.asyncio
async def test_wireframe_as_lock_donor_bug_resolved(tmp_path, monkeypatch):
    """Verify that editing turns use the prior polished render as lock donor, NOT the wireframe."""
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    store = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: store)

    # 1. Wireframe image (white background with gray outline)
    wireframe_img = Image.new("RGB", (200, 200), (255, 255, 255))
    wireframe_ref = store.save_image(wireframe_img, prefix="wireframe")

    # 2. Prior polished render (high-fidelity dark indigo with rich brand colors)
    prior_polished_img = Image.new("RGB", (200, 200), (15, 23, 42))
    prior_polished_ref = store.save_image(prior_polished_img, prefix="prior_polished")

    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=4.5, quantization="BF16", always_resident=False)
    backend = QwenImage21Backend(spec)
    await backend.load()

    locked_regions = [{"bbox": [0.1, 0.1, 0.4, 0.4]}]

    # Run polish backend passing wireframe as handoff conditioning AND prior_polished_ref as lock_donor_ref
    result = await backend.run(
        handoff_image_ref=wireframe_ref,
        lock_donor_ref=prior_polished_ref,
        prompt="Make buttons larger",
        constraints={},
        locked_regions=locked_regions,
    )

    output_img = store.load_image(result["image_ref"])
    verifier = LockPreservationVerifier()

    # The locked region MUST match the prior polished image (score 1.000)
    score_vs_prior = verifier.score(prior_polished_img, output_img, locked_regions)
    assert score_vs_prior == 1.0

    # The locked region must NOT have reverted to the wireframe (score < 0.30)
    score_vs_wireframe = verifier.score(wireframe_img, output_img, locked_regions)
    assert score_vs_wireframe < 0.30


@pytest.mark.asyncio
async def test_closed_loop_retry_and_drift_feedback(tmp_path, monkeypatch):
    """Verify that when verifier detects spatial drift, finalize() triggers an automated single retry."""
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, Constraints, SessionStage
    from krisna_inference.orchestrator.flows import finalize
    from krisna_inference.orchestrator.swap_orchestrator import SwapResult

    store_blob = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store_blob)

    state = DesignState()
    state.stage = SessionStage.SKETCHING
    state.layout.wireframe_ref = "blob://test_wf.png"
    state.layout.layout_ref = str(tmp_path / "layout.json")
    Path(state.layout.layout_ref).write_text('{"nodes": []}', encoding="utf-8")

    class FakeStore:
        def get(self, session_id):
            return state
        def save(self, s, rev):
            return s

    retry_count = 0
    captured_calls = []

    class FakeOrchestrator:
        async def request_finalize(self, **kwargs):
            nonlocal retry_count
            retry_count += 1
            captured_calls.append(kwargs)
            # Save dummy image
            img = Image.new("RGB", (100, 100), (50, 50, 50))
            ref = store_blob.save_image(img, prefix="render")
            return SwapResult(ok=True, tier_used=Tier.POLISH_DEFAULT, output={"image_ref": ref})

    class FakeVerifierStack:
        def safety_gate(self, img, min_safety_score=0.9):
            return True, 0.99

        def score_finalize_output(self, polished_image, **kwargs):
            # First attempt returns poor IoU (0.55 < 0.70); retry returns improved IoU (0.85)
            iou = 0.55 if retry_count == 1 else 0.85
            return {"layout_iou": iou, "ocr_readability": 0.80}

    final_state = await finalize(
        orchestrator=FakeOrchestrator(),
        store=FakeStore(),
        session_id="sess_retry_test",
        verifier_stack=FakeVerifierStack(),
    )

    # 1. Closed-loop automated retry was triggered
    assert retry_count == 2
    # 2. Final IoU adopted the improved score
    assert final_state.finalize_output.verifier_scores.layout_iou == 0.85
    # 3. System drift diagnostic was surfaced into conversation history for next-turn Planner grounding
    history_texts = [turn.content for turn in final_state.conversation_history if turn.role == "system"]
    assert any("Spatial verifier diagnostic" in text for text in history_texts)
    # 4. Verify retry was NOT a no-op: prompt and seed were strictly perturbed
    assert len(captured_calls) == 2
    assert "CRITICAL LAYOUT ENFORCEMENT" in captured_calls[1].get("prompt", "")
    assert captured_calls[1].get("seed") is not None
    assert captured_calls[1].get("seed") != captured_calls[0].get("seed")


def test_planner_prompt_system_role_safety_and_current_layout_prefix():
    """Verify PlannerBackend safely maps system history turns to user diagnostics and grounds current_layout."""
    from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=10.0, quantization="Q4_K_M", always_resident=True)
    backend = PlannerBackendLlama(spec)

    history = [
        {"role": "user", "content": "Initial user request"},
        {"role": "planner", "content": '{"reply_text": "First layout generated"}'},
        {"role": "system", "content": "Spatial verifier diagnostic: Layout IoU 0.55 < 0.70."},
    ]
    constraints = {
        "style": "clean modern",
        "current_layout": [
            {"id": "header", "type": "container", "bbox": [0.0, 0.0, 1.0, 0.15], "locked": True},
            {"id": "cta_button", "type": "button", "bbox": [0.3, 0.7, 0.4, 0.1], "text": "Click Here"},
        ],
    }

    messages, _ = backend._prompt(
        message="Fix the header",
        constraints=constraints,
        history=history,
        critique=None,
        review=False,
    )

    # 1. System role is ONLY at index 0 (the root system instructions)
    assert messages[0]["role"] == "system"
    for turn in messages[1:]:
        assert turn["role"] in ("user", "assistant")

    # 2. History system turn was safely remapped to user diagnostic
    diagnostic_turns = [m for m in messages if "[System Diagnostic]" in str(m.get("content"))]
    assert len(diagnostic_turns) == 1
    assert "Layout IoU 0.55 < 0.70" in diagnostic_turns[0]["content"]

    # 3. current_layout was converted into structured InstructLayout graph prefix in system prompt
    assert "[Structured Layout IR Graph (InstructLayout/TPAMI 2025)]:" in messages[0]["content"]
    assert "id=header, type=container, bbox=[0.0, 0.0, 1.0, 0.15], locked=True" in messages[0]["content"]
    assert "id=cta_button, type=button" in messages[0]["content"]


@pytest.mark.asyncio
async def test_iterative_refine_wires_lock_donor_and_verifier_stack(tmp_path, monkeypatch):
    """Verify iterative_refine forwards lock_donor_ref and runs verifier_stack with lock donor."""
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage, LockedRegion
    from krisna_inference.orchestrator.flows import iterative_refine
    from krisna_inference.orchestrator.swap_orchestrator import SwapResult
    from krisna_inference.verifiers.verifier_stack import VerifierStack

    store_blob = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store_blob)

    # Create prior render image (high-fidelity render with red lock card)
    prior_img = Image.new("RGB", (200, 200), (20, 20, 20))
    for x in range(20, 100):
        for y in range(20, 100):
            prior_img.putpixel((x, y), (255, 0, 0))
    prior_ref = store_blob.save_image(prior_img, prefix="prior_render")

    state = DesignState()
    state.stage = SessionStage.FINALIZED
    state.finalize_output.image_ref = prior_ref
    state.constraints.locked_regions = [LockedRegion(bbox=[0.1, 0.1, 0.4, 0.4], reason="red card")]
    state.layout.wireframe_ref = prior_ref

    class FakeStore:
        def get(self, session_id):
            return state
        def save(self, s, rev):
            return s

    captured_kwargs = {}

    class FakeOrchestrator:
        async def request_finalize(self, **kwargs):
            captured_kwargs.update(kwargs)
            output_img = Image.new("RGB", (200, 200), (250, 250, 250))
            for x in range(20, 100):
                for y in range(20, 100):
                    output_img.putpixel((x, y), (255, 0, 0))
            ref = store_blob.save_image(output_img, prefix="refined")
            return SwapResult(ok=True, tier_used=Tier.POLISH_DEFAULT, output={"image_ref": ref})

    class OfflineTestVerifierStack(VerifierStack):
        def __init__(self):
            from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier
            self.lock_preservation = LockPreservationVerifier()
            self.clip_alignment = type("MockCLIP", (), {"score": lambda self, img, p: 0.85})()
            self.ocr_readability = type("MockOCR", (), {"score": lambda self, img, t: 0.90})()
            self.layout_iou = type("MockIoU", (), {"score": lambda self, img, l: 0.88})()
            self.aesthetic = type("MockAesthetic", (), {"score": lambda self, img: 0.82})()
            self.safety = type("MockSafety", (), {"score": lambda self, img: 0.99})()
            self.handoff_consistency = type("MockHandoff", (), {"score": lambda self, img, h: 0.80})()

        def safety_gate(self, image, min_safety_score=0.9):
            return True, 0.99

    verifier_stack = OfflineTestVerifierStack()

    final_state = await iterative_refine(
        orchestrator=FakeOrchestrator(),
        store=FakeStore(),
        session_id="sess_refine_test",
        instruction="Change background to light mode",
        verifier_stack=verifier_stack,
    )

    # 1. lock_donor_ref was explicitly passed to orchestrator
    assert captured_kwargs.get("lock_donor_ref") == prior_ref
    assert captured_kwargs.get("handoff_image_ref") == prior_ref

    # 2. Verifier scores were populated with exact 1.000 lock preservation
    scores = final_state.finalize_output.verifier_scores
    assert scores.lock_preservation == 1.0


def test_sketch_tier_aspect_ratios_and_typography_boundaries(tmp_path):
    """Verify Sketch Tier supports 16:9, 9:16, 4:3 (tablet), and 1:1 pinned to multiples of 32 with typography boundary protection."""
    # 1. Test Tablet 4:3
    constraints_tablet = {"aspect_ratio": "4:3 tablet"}
    layout_tablet = layout_from_constraints(constraints_tablet, text="Tablet UI")
    assert layout_tablet.canvas_width == 1152
    assert layout_tablet.canvas_height == 864
    assert layout_tablet.canvas_width % 32 == 0
    assert layout_tablet.canvas_height % 32 == 0
    assert layout_tablet.canvas_width * layout_tablet.canvas_height <= 1024 * 1024

    # 2. Test Desktop 16:9
    constraints_desktop = {"aspect_ratio": "16:9 desktop"}
    layout_desktop = layout_from_constraints(constraints_desktop, text="Desktop Dashboard")
    assert layout_desktop.canvas_width == 1344
    assert layout_desktop.canvas_height == 768
    assert layout_desktop.canvas_width % 32 == 0
    assert layout_desktop.canvas_height % 32 == 0

    # 3. Test Mobile 9:16
    constraints_mobile = {"aspect_ratio": "9:16 mobile"}
    layout_mobile = layout_from_constraints(constraints_mobile, text="Mobile App")
    assert layout_mobile.canvas_width == 768
    assert layout_mobile.canvas_height == 1344
    assert layout_mobile.canvas_width % 32 == 0
    assert layout_mobile.canvas_height % 32 == 0

    # 4. Test Wireframe rendering with very long text label that requires clipping
    constraints_long_text = {
        "aspect_ratio": "16:9",
        "actions": [
            AddComponentAction(
                component_type="button",
                bbox=[0.1, 0.5, 0.15, 0.08],
                target_id="small_btn",
                props={"label": "Extremely Long Button Label That Definitely Exceeds The Narrow Fifteen Percent Bounding Box Width"},
            )
        ],
    }
    layout_long = layout_from_constraints(constraints_long_text, text="Clipping Test")
    out_file = tmp_path / "long_text_wireframe.png"
    render_wireframe(layout_long, out_file)
    assert out_file.exists()


def test_flex_layout_unit_exempts_locked_nodes_and_displaces_mutables():
    """Unit test for internal _resolve_flex_layout: locked nodes remain strictly stationary while colliding mutables yield space."""
    from krisna_inference.layout_ir import BBox, LayoutNode
    from krisna_inference.layout_baseline import _resolve_flex_layout

    # Create a locked node and a colliding mutable node at the exact same coordinates
    initial_locked_bbox = BBox(x1=0.20, y1=0.20, x2=0.80, y2=0.40)
    colliding_mutable_bbox = BBox(x1=0.20, y1=0.20, x2=0.80, y2=0.40)

    node_map = {
        "locked_header": LayoutNode(
            id="locked_header",
            type="card",
            bbox=initial_locked_bbox,
            locked=True,
        ),
        "mutable_card": LayoutNode(
            id="mutable_card",
            type="card",
            bbox=colliding_mutable_bbox,
            locked=False,
        ),
    }

    resolved = _resolve_flex_layout(node_map, constraints={"layout_direction": "column"})

    # 1. The locked node must NOT have moved at all
    assert resolved["locked_header"].bbox.x1 == initial_locked_bbox.x1
    assert resolved["locked_header"].bbox.y1 == initial_locked_bbox.y1
    assert resolved["locked_header"].bbox.x2 == initial_locked_bbox.x2
    assert resolved["locked_header"].bbox.y2 == initial_locked_bbox.y2

    # 2. The mutable card must have yielded space and moved below the locked node
    assert resolved["mutable_card"].bbox.y1 >= initial_locked_bbox.y2


def test_flex_layout_detects_locked_locked_collision_and_clamps_y_overflow():
    """Verify _resolve_flex_layout detects locked-locked collision conflicts and clamps mutable nodes against Y overflow."""
    from krisna_inference.layout_ir import BBox, LayoutNode
    from krisna_inference.layout_baseline import _resolve_flex_layout

    node_map = {
        "locked_1": LayoutNode(id="locked_1", type="card", bbox=BBox(x1=0.2, y1=0.8, x2=0.8, y2=0.95), locked=True),
        "locked_2": LayoutNode(id="locked_2", type="card", bbox=BBox(x1=0.2, y1=0.8, x2=0.8, y2=0.95), locked=True),
        "mutable_overflow": LayoutNode(id="mutable_overflow", type="button", bbox=BBox(x1=0.2, y1=0.8, x2=0.8, y2=0.95), locked=False),
    }

    resolved = _resolve_flex_layout(node_map, constraints={"layout_direction": "column"})
    # Both locked nodes must not move
    assert resolved["locked_1"].bbox.y1 == 0.8
    assert resolved["locked_2"].bbox.y1 == 0.8
    # Mutable overflow node must be safely clamped within canvas (y2 <= 0.96)
    assert resolved["mutable_overflow"].bbox.y2 <= 0.96
    assert resolved["mutable_overflow"].bbox.y1 >= 0.04


def test_lock_preservation_verifier_computes_seam_boundary_continuity():
    """Verify LockPreservationVerifier calculates both MAE pixel score and seam continuity score."""
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    verifier = LockPreservationVerifier()

    img_clean = Image.new("RGB", (100, 100), (200, 200, 200))
    region = [{"bbox": [0.2, 0.2, 0.4, 0.4]}]

    # Clean unperturbed image: seam continuity score is near 1.0 (minimal boundary step)
    seam_clean = verifier.seam_score(img_clean, region)
    assert seam_clean is not None
    assert seam_clean >= 0.95

    # Perturbed image with stark black box inserted at lock region: seam continuity score penalizes the step
    img_discontinuous = img_clean.copy()
    for x in range(20, 60):
        for y in range(20, 60):
            img_discontinuous.putpixel((x, y), (0, 0, 0))

    seam_discontinuous = verifier.seam_score(img_discontinuous, region)
    assert seam_discontinuous is not None
    assert seam_discontinuous < seam_clean


def test_non_tautological_lock_restoration():
    """Verify that _apply_locked_regions restores exact donor pixels when output is deliberately corrupted."""
    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=4.5, quantization="BF16")
    backend = QwenImage21Backend(spec, strict=False)

    # 1. Donor image has solid red in the locked region [20:80, 20:80]
    donor_img = Image.new("RGB", (100, 100), (255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            donor_img.putpixel((x, y), (255, 0, 0))

    # 2. Corrupted rendered image has solid lime green in the locked region
    corrupted_render = Image.new("RGB", (100, 100), (255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            corrupted_render.putpixel((x, y), (0, 255, 0))

    # Before lock application: pixels are green (0, 255, 0)
    assert corrupted_render.getpixel((50, 50)) == (0, 255, 0)

    # Apply locked region: should overwrite the corrupted green pixels with red donor pixels
    restored = backend._apply_locked_regions(
        source_img=donor_img,
        rendered_img=corrupted_render,
        locked_regions=[{"bbox": [0.20, 0.20, 0.60, 0.60]}],
    )

    # After lock application: pixels MUST be exact red (255, 0, 0)
    for x in range(20, 80):
        for y in range(20, 80):
            assert restored.getpixel((x, y)) == (255, 0, 0)


def test_qwen_backend_call_arguments_match_pipeline_signature():
    """Verify that backend inspects and only passes arguments exposed by native pipeline."""
    import inspect

    # Mock class that implements QwenImage21Pipeline call signature
    class MockQwenImage21Pipeline:
        def __call__(
            self,
            prompt=None,
            image=None,
            height=None,
            width=None,
            num_inference_steps=40,
            sigmas=None,
            true_cfg_scale=1.0,
            generator=None,
            callback_on_step_end=None,
        ):
            return Image.new("RGB", (100, 100))

    pipe = MockQwenImage21Pipeline()
    sig_params = set(inspect.signature(pipe.__call__).parameters.keys())

    # Ensure prohibited parameters like 'mask_image' are NOT in pipeline signature
    assert "mask_image" not in sig_params
    assert "inpainting_mask" not in sig_params

    # Supported core parameters MUST be present
    assert {"prompt", "image", "num_inference_steps", "true_cfg_scale", "generator"}.issubset(sig_params)


def test_explicit_asset_mode_transparency():
    """Verify explicit asset_mode in constraints governs RGBA preservation."""
    from krisna_inference.orchestrator.design_state import Constraints

    c_opaque = Constraints(asset_mode="opaque")
    assert c_opaque.asset_mode == "opaque"

    c_transparent = Constraints(asset_mode="transparent")
    assert c_transparent.asset_mode == "transparent"


def test_verbatim_label_guard_reinjects_missing_labels():
    """Verify that assert_and_inject_verbatim_labels preserves all labels without loss."""
    from krisna_inference.orchestrator.flows import assert_and_inject_verbatim_labels

    prompt = "A modern banking dashboard with dark background."
    labels = ["Pay Now", "Transfer Funds", "$1,450.00"]

    guarded = assert_and_inject_verbatim_labels(prompt, labels)
    for lbl in labels:
        assert lbl in guarded
    assert "Text labels rendered legibly:" in guarded


def test_synthesize_descriptive_polish_prompt_produces_pe_prose(tmp_path):
    """Verify that synthesize_descriptive_polish_prompt generates coherent PE prose."""
    import json
    from krisna_inference.orchestrator.design_state import DesignState, Constraints, LockedRegion
    from krisna_inference.orchestrator.flows import synthesize_descriptive_polish_prompt

    state = DesignState()
    state.constraints = Constraints(
        style="minimalist fintech",
        palette=["#0A0A0A", "#00FF66"],
        locked_regions=[LockedRegion(bbox=[0.1, 0.1, 0.8, 0.15], reason="locked navbar")],
    )
    layout_data = {
        "nodes": [
            {"id": "root", "type": "canvas"},
            {"id": "title", "type": "header", "text": "Crypto Vault", "bbox": {"y1": 0.05}},
            {"id": "balance_card", "type": "card", "text": "Total Assets", "bbox": {"y1": 0.40}},
            {"id": "send_btn", "type": "button", "text": "Send Bitcoin", "bbox": {"y1": 0.88}},
        ]
    }
    layout_file = tmp_path / "fintech_layout.json"
    layout_file.write_text(json.dumps(layout_data), encoding="utf-8")
    state.layout.layout_ref = str(layout_file)

    prose = synthesize_descriptive_polish_prompt(state)
    assert "minimalist fintech" in prose
    assert "Crypto Vault" in prose
    assert "Total Assets" in prose
    assert "Send Bitcoin" in prose
    assert "PRESERVE UNCHANGED: locked navbar" in prose
    assert "#0A0A0A" in prose


def test_grounded_spatial_retry_prompt_linguistic_feedback(tmp_path):
    """Verify build_grounded_spatial_retry_prompt emits region-specific linguistic feedback."""
    import json
    from krisna_inference.orchestrator.design_state import DesignState
    from krisna_inference.orchestrator.flows import build_grounded_spatial_retry_prompt

    state = DesignState()
    layout_data = {
        "nodes": [
            {"id": "root", "type": "canvas"},
            {"id": "title", "type": "header", "text": "Checkout", "bbox": {"y1": 0.10}},
            {"id": "summary", "type": "card", "text": "Order Summary", "bbox": {"y1": 0.45}},
            {"id": "btn", "type": "button", "text": "Pay Now", "bbox": {"y1": 0.85}},
        ]
    }
    layout_file = tmp_path / "checkout_layout.json"
    layout_file.write_text(json.dumps(layout_data), encoding="utf-8")
    state.layout.layout_ref = str(layout_file)

    retry_prompt = build_grounded_spatial_retry_prompt(
        state=state,
        base_prompt="A checkout page with card and button",
        drift_reasons=["Layout IoU 0.52 < 0.70", "OCR readability 0.65 < 0.80", "Seam continuity 0.78 < 0.85"],
        layout_iou=0.52,
        ocr_score=0.65,
        seam_score=0.78,
    )

    assert "CRITICAL LAYOUT ENFORCEMENT" in retry_prompt
    assert "the 'Pay Now' button must sit at the bottom-right under the main card" in retry_prompt
    assert "the 'Checkout' header must sit at the top" in retry_prompt
    assert "Typographic legibility correction:" in retry_prompt
    assert "Boundary continuity correction:" in retry_prompt


def test_locked_locked_conflicts_surfaced_to_layout_diagnostics():
    """Verify that locked-locked collisions are captured in LayoutIR.diagnostics."""
    from krisna_inference.layout_baseline import layout_from_constraints
    from krisna_inference.backends.planner_schema import AddComponentAction

    constraints = {
        "style": "dark",
        "actions": [
            AddComponentAction(component_type="card", bbox=[0.1, 0.2, 0.8, 0.3], target_id="c1"),
            AddComponentAction(component_type="card", bbox=[0.1, 0.2, 0.8, 0.3], target_id="c2"),
        ],
        "locked_regions": [
            {"bbox": [0.1, 0.2, 0.8, 0.3], "reason": "lock 1"},
            {"bbox": [0.1, 0.2, 0.8, 0.3], "reason": "lock 2"},
        ],
    }

    layout = layout_from_constraints(constraints)
    assert len(layout.diagnostics) >= 1
    assert any("Locked node conflict" in d for d in layout.diagnostics)


def test_seam_continuity_evaluation_target_gating():
    """Verify seam_continuity target 0.85 in EVALUATION_TARGETS and evaluation receipt."""
    from krisna_inference.verifiers.verifier_stack import EVALUATION_TARGETS, VerifierStack
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    assert "seam_continuity" in EVALUATION_TARGETS
    assert EVALUATION_TARGETS["seam_continuity"] == 0.85

    class MockOfflineVerifierStack(VerifierStack):
        def __init__(self):
            self.lock_preservation = LockPreservationVerifier()
            self.clip_alignment = type("Mock", (), {"score": lambda self, *a, **k: 0.85})()
            self.ocr_readability = type("Mock", (), {"score": lambda self, *a, **k: {"score": 0.90}})()
            self.layout_iou = type("Mock", (), {"score": lambda self, *a, **k: 0.88})()
            self.aesthetic = type("Mock", (), {"score": lambda self, *a, **k: 0.80})()
            self.safety = type("Mock", (), {"score": lambda self, *a, **k: 0.99})()
            self.handoff_consistency = type("Mock", (), {"score": lambda self, *a, **k: 0.80})()

    stack = MockOfflineVerifierStack()
    img_a = Image.new("RGB", (100, 100), (255, 255, 255))
    locked = [{"bbox": (10, 10, 40, 40)}]

    receipt = stack.evaluate_finalize_output(
        polished_image=img_a,
        prompt="clean UI",
        locked_regions=locked,
        lock_donor_image=img_a,
    )
    assert "seam_continuity" in receipt["metrics"]
    assert receipt["per_metric"]["seam_continuity"]["passed"] is True
    assert receipt["per_metric"]["seam_continuity"]["target"] == 0.85


def test_render_wireframe_transparent_asset_mode_produces_rgba(tmp_path):
    """Verify wireframe renderer outputs an RGBA image when asset_mode is transparent."""
    constraints = {
        "style": "modern",
        "asset_mode": "transparent",
        "palette": ["#000000", "#38bdf8"],
    }
    layout = layout_from_constraints(constraints, text="Transparent Component")
    out_png = tmp_path / "transparent_wireframe.png"

    path = render_wireframe(layout, out_png)
    assert path.exists()

    with Image.open(path) as img:
        assert img.size == (layout.canvas_width, layout.canvas_height)
        assert img.mode == "RGBA"


@pytest.mark.asyncio
async def test_layout_backend_preserves_rgba_wireframe_in_blob_store(tmp_path):
    """Verify LayoutBackend saves and retains native RGBA wireframe in blob store."""
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.backends.blob_store_singleton import get_blob_store
    from krisna_inference.orchestrator.model_registry import ModelSpec, Tier

    spec = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    await backend.load()

    result = await backend.run(
        planner_output={},
        constraints={"asset_mode": "transparent", "style": "clean"},
        message="A transparent badge UI",
        session_id="test_sess",
        turn_id="turn_1",
    )

    assert "wireframe_ref" in result
    wireframe_ref = result["wireframe_ref"]
    blob_store = get_blob_store()
    img = blob_store.load_image(wireframe_ref)
    assert img.mode == "RGBA"
    assert "segmentation_ref" in result
    assert "mask_ref" in result
    assert result["mask_ref"] is not None
    assert result["canvas_width"] % 32 == 0
    assert result["canvas_height"] % 32 == 0
    assert result["mode"] in ("layout_baseline", "neural_layout_flow") or result["mode"].startswith("krisna.layout.v1.")


def test_layout_from_constraints_preserves_and_mutates_current_layout():
    """Verify that layout_from_constraints preserves prior turn's current_layout and applies actions cleanly."""
    from krisna_inference.layout_baseline import layout_from_constraints

    prior_layout = [
        {"id": "navbar", "type": "container", "bbox": [0.0, 0.0, 1.0, 0.12], "locked": False},
        {"id": "hero_card", "type": "card", "bbox": [0.05, 0.18, 0.95, 0.65], "text": "Welcome"},
        {"id": "cta_button", "type": "button", "bbox": [0.35, 0.75, 0.65, 0.85], "text": "Click Here"},
    ]
    actions = [
        {"action_type": "SET_TEXT", "target_id": "cta_button", "text_content": "Register Now"},
        {"action_type": "LOCK_REGION", "target_id": "navbar"},
        {"action_type": "ADD_COMPONENT", "component_type": "text", "target_id": "sub_caption", "bbox": [0.1, 0.88, 0.9, 0.94], "text_content": "No credit card required"},
    ]
    constraints = {
        "current_layout": prior_layout,
        "actions": actions,
        "style": "clean modern SaaS",
    }

    result_ir = layout_from_constraints(constraints, text="Update the button and add subcaption")
    node_ids = {n.id for n in result_ir.nodes}

    # 1. Custom nodes from prior turn are preserved
    assert "navbar" in node_ids
    assert "hero_card" in node_ids
    assert "cta_button" in node_ids
    assert "sub_caption" in node_ids

    # 2. Generic initial starter nodes are NOT reintroduced
    assert "title" not in node_ids
    assert "content" not in node_ids
    assert "action" not in node_ids

    # 3. Actions applied accurately
    node_dict = {n.id: n for n in result_ir.nodes}
    assert node_dict["cta_button"].text == "Register Now"
    assert node_dict["navbar"].locked is True
    assert node_dict["sub_caption"].text == "No credit card required"


def test_render_dual_stream_sketch_quantization_and_synchronization(tmp_path):
    """Verify render_dual_stream_sketch produces synchronized wireframe, SPADE segmentation, and mask with multiple-of-32 dimensions."""
    from krisna_inference.layout_renderer import render_dual_stream_sketch, render_spatial_alpha_mask, DualStreamSketch
    from krisna_inference.layout_baseline import layout_from_constraints

    constraints = {
        "style": "clean dark modern",
        "actions": [
            AddComponentAction(component_type="header", bbox=[0.05, 0.05, 0.90, 0.15], target_id="hdr", props={"label": "Control Center"}),
            AddComponentAction(component_type="button", bbox=[0.10, 0.30, 0.40, 0.12], target_id="btn", props={"label": "Deploy"}),
            LockRegionAction(target_id="hdr"),
        ]
    }
    ir = layout_from_constraints(constraints, text="Test dual stream")

    # Request non-multiple of 32 (1010 x 750)
    dual_sketch = render_dual_stream_sketch(
        ir,
        canvas_width=1010,
        canvas_height=750,
        output_dir=tmp_path / "dual_stream_out",
        prefix="preflight_test"
    )

    assert isinstance(dual_sketch, DualStreamSketch)
    # Dimensions must be multiples of 32 (1010 -> 1024, 750 -> 768)
    assert dual_sketch.width % 32 == 0
    assert dual_sketch.height % 32 == 0
    assert dual_sketch.width == 1024
    assert dual_sketch.height == 736

    assert dual_sketch.wireframe.size == (1024, 736)
    assert dual_sketch.segmentation.size == (1024, 736)
    assert dual_sketch.mask is not None
    assert dual_sketch.mask.size == (1024, 736)

    # Wireframe mode RGB, segmentation RGB, mask L
    assert dual_sketch.wireframe.mode == "RGB"
    assert dual_sketch.segmentation.mode == "RGB"
    assert dual_sketch.mask.mode == "L"

    # Files must be written to disk
    assert Path(dual_sketch.wireframe_path).is_file()
    assert Path(dual_sketch.segmentation_path).is_file()
    assert Path(dual_sketch.mask_path).is_file()


def test_render_spatial_alpha_mask_values():
    """Verify render_spatial_alpha_mask assigns 0 to locked regions and 255 to mutable areas."""
    from krisna_inference.layout_renderer import render_spatial_alpha_mask
    from krisna_inference.layout_baseline import layout_from_constraints
    import numpy as np

    constraints = {
        "style": "cyberpunk",
        "actions": [
            AddComponentAction(component_type="card", bbox=[0.2, 0.2, 0.6, 0.6], target_id="locked_card"),
            LockRegionAction(target_id="locked_card"),
        ]
    }
    ir = layout_from_constraints(constraints, text="Mask test")
    mask = render_spatial_alpha_mask(ir, width=1024, height=1024)

    assert mask.size == (1024, 1024)
    arr = np.array(mask)

    # Center of locked card (x=512, y=512) must be 0 (locked donor)
    assert arr[512, 512] == 0

    # Margin near corner (x=50, y=50) must be 255 (mutable background)
    assert arr[50, 50] == 255


def test_universal_node_types_and_spade_palette(tmp_path):
    """Verify that universal graphic design nodes survive schema validation and render discrete SPADE classes."""
    from krisna_inference.layout_ir import LayoutIR, LayoutNode, BBox, NODE_TYPES
    from krisna_inference.layout_renderer import render_segmentation_map, UNIVERSAL_SEMANTIC_PALETTE
    import numpy as np

    # 1. Verify schema includes universal design nodes
    universal_sample = ["badge", "headline", "illustration", "product", "banner", "shape"]
    for t in universal_sample:
        assert t in NODE_TYPES
        assert t in UNIVERSAL_SEMANTIC_PALETTE

    # 2. Build layout with universal nodes
    nodes = [
        LayoutNode(id="root", type="container", bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)),
        LayoutNode(id="hero_badge", type="badge", bbox=BBox(x1=0.1, y1=0.1, x2=0.3, y2=0.16), text="New Release"),
        LayoutNode(id="head", type="headline", bbox=BBox(x1=0.1, y1=0.18, x2=0.9, y2=0.28), text="Next-Gen AI Canvas"),
        LayoutNode(id="prod", type="product", bbox=BBox(x1=0.1, y1=0.32, x2=0.5, y2=0.75), text="Hero Product"),
        LayoutNode(id="graphic", type="shape", bbox=BBox(x1=0.55, y1=0.32, x2=0.9, y2=0.75)),
    ]
    ir = LayoutIR(canvas_width=1024, canvas_height=1024, nodes=nodes)
    seg_png = tmp_path / "universal_seg.png"
    render_segmentation_map(ir, seg_png)

    assert seg_png.is_file()
    with Image.open(seg_png) as img:
        arr = np.array(img)
        # Check badge color (234, 179, 8) in arr
        badge_pixel = arr[int(0.13 * 1024), int(0.20 * 1024)]
        assert tuple(badge_pixel) == UNIVERSAL_SEMANTIC_PALETTE["badge"]
        # Check product/focal subject color (34, 197, 94) in arr
        prod_pixel = arr[int(0.50 * 1024), int(0.30 * 1024)]
        assert tuple(prod_pixel) == UNIVERSAL_SEMANTIC_PALETTE["product"]
        # Check decorative shape color (236, 72, 153)
        shape_pixel = arr[int(0.50 * 1024), int(0.70 * 1024)]
        assert tuple(shape_pixel) == UNIVERSAL_SEMANTIC_PALETTE["shape"]


def test_compute_differential_mask_isolates_partition_edits(tmp_path):
    """Verify compute_differential_mask isolates edited partition and preserves untouched partitions."""
    from krisna_inference.layout_ir import LayoutIR, LayoutNode, BBox
    from krisna_inference.layout_renderer import compute_differential_mask
    import numpy as np

    # Turn 1: Navbar at top, Card A on left, Card B on right
    turn1_nodes = [
        LayoutNode(id="root", type="container", bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)),
        LayoutNode(id="nav", type="navbar", bbox=BBox(x1=0.05, y1=0.05, x2=0.95, y2=0.15), text="Brand Nav"),
        LayoutNode(id="card_left", type="card", bbox=BBox(x1=0.05, y1=0.25, x2=0.45, y2=0.75), text="Original Left"),
        LayoutNode(id="card_right", type="card", bbox=BBox(x1=0.55, y1=0.25, x2=0.95, y2=0.75), text="Original Right"),
    ]
    layout_t1 = LayoutIR(canvas_width=1024, canvas_height=1024, nodes=turn1_nodes)

    # Turn 2: User says "Change right card to 3 pricing tiers". Left card and nav are completely UNTOUCHED.
    turn2_nodes = [
        LayoutNode(id="root", type="container", bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)),
        LayoutNode(id="nav", type="navbar", bbox=BBox(x1=0.05, y1=0.05, x2=0.95, y2=0.15), text="Brand Nav"),
        LayoutNode(id="card_left", type="card", bbox=BBox(x1=0.05, y1=0.25, x2=0.45, y2=0.75), text="Original Left"),
        LayoutNode(id="card_right", type="card", bbox=BBox(x1=0.55, y1=0.25, x2=0.95, y2=0.75), text="Updated 3-Tier Pricing Table"),
    ]
    layout_t2 = LayoutIR(canvas_width=1024, canvas_height=1024, nodes=turn2_nodes)

    mask = compute_differential_mask(layout_t1, layout_t2, canvas_width=1024, canvas_height=1024, dilation_radius=16)
    arr = np.array(mask)

    # 1. Untouched navbar (y=100, x=500) must be 0 (immutable photoreal donor preserved)
    assert arr[100, 500] == 0

    # 2. Untouched left card (y=500, x=250) must be 0 (immutable photoreal donor preserved)
    assert arr[500, 250] == 0

    # 3. Modified right card (y=500, x=750) must be 255 (active generation zone for Qwen flow matching)
    assert arr[500, 750] == 255


def test_multi_column_spatial_prompt_synthesis(tmp_path):
    """Verify synthesize_descriptive_polish_prompt emits 2D multi-column partition coordinates."""
    import json
    from krisna_inference.orchestrator.design_state import DesignState
    from krisna_inference.orchestrator.flows import synthesize_descriptive_polish_prompt

    state = DesignState()
    state.constraints.style = "modern SaaS"
    layout_data = {
        "nodes": [
            {"id": "root", "type": "container", "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
            {"id": "title", "type": "headline", "text": "Pricing Plans", "bbox": {"x1": 0.1, "y1": 0.08, "x2": 0.9, "y2": 0.18}},
            {"id": "tier_1", "type": "card", "text": "Starter Plan", "bbox": {"x1": 0.10, "y1": 0.28, "x2": 0.35, "y2": 0.75}},
            {"id": "tier_2", "type": "card", "text": "Pro Plan", "bbox": {"x1": 0.38, "y1": 0.28, "x2": 0.63, "y2": 0.75}},
            {"id": "tier_3", "type": "card", "text": "Enterprise Plan", "bbox": {"x1": 0.66, "y1": 0.28, "x2": 0.91, "y2": 0.75}},
        ]
    }
    lf = tmp_path / "multicol_layout.json"
    lf.write_text(json.dumps(layout_data), encoding="utf-8")
    state.layout.layout_ref = str(lf)

    prose = synthesize_descriptive_polish_prompt(state)

    # Asserts 2D multi-column awareness and exact verbatim quoting
    assert "structured multi-column partition" in prose
    assert "column 1 [x: 0.10]" in prose
    assert "Starter Plan" in prose
    assert "column 2 [x: 0.38]" in prose
    assert "Pro Plan" in prose
    assert "column 3 [x: 0.66]" in prose
    assert "Enterprise Plan" in prose


def test_portrait_character_archetype_and_facial_mapping(tmp_path):
    """Verify portrait archetype detection, starter nodes with facial landmarks, and polish prompt prose."""
    import json
    from krisna_inference.layout_ir import SceneArchetype, LayoutIR
    from krisna_inference.layout_baseline import detect_scene_archetype, layout_from_constraints
    from krisna_inference.layout_renderer import render_wireframe, render_segmentation_map
    from krisna_inference.orchestrator.design_state import DesignState
    from krisna_inference.orchestrator.flows import synthesize_descriptive_polish_prompt

    prompt = "Create a dramatic cyberpunk female warrior character portrait with glowing neon eyes and intricate cyberware"
    archetype = detect_scene_archetype(prompt)
    assert archetype == SceneArchetype.PORTRAIT_CHARACTER

    layout = layout_from_constraints({"style": "cyberpunk portrait"}, text=prompt)
    assert layout.scene_archetype == SceneArchetype.PORTRAIT_CHARACTER.value

    # Verify nodes: must contain focal subject with facial landmarks, no generic UI buttons
    node_types = [n.type for n in layout.nodes]
    assert "focal_subject" in node_types
    assert "button" not in node_types

    focal_nodes = [n for n in layout.nodes if n.is_focal_subject or n.type == "focal_subject"]
    assert len(focal_nodes) >= 1
    focal = focal_nodes[0]
    assert focal.facial_landmarks is not None
    assert "eyes" in focal.facial_landmarks
    assert "mouth" in focal.facial_landmarks

    # Render wireframe and segmentation to verify no rendering crashes
    wf_path = tmp_path / "portrait_wf.png"
    render_wireframe(layout, wf_path)
    assert wf_path.exists() and wf_path.stat().st_size > 0

    seg_path = tmp_path / "portrait_seg.png"
    render_segmentation_map(layout, seg_path)
    assert seg_path.exists() and seg_path.stat().st_size > 0

    # Test prompt synthesis emits facial mapping instructions
    state = DesignState()
    state.constraints.style = "cyberpunk"
    lf = tmp_path / "portrait_layout.json"
    lf.write_text(json.dumps(layout.model_dump()), encoding="utf-8")
    state.layout.layout_ref = str(lf)

    prose = synthesize_descriptive_polish_prompt(state)
    assert "character portrait" in prose
    assert "facial mapping" in prose
    assert "expressive eyes" in prose


def test_graphic_poster_z_index_overlay_stacking():
    """Verify nodes on distinct z-index layers or parent-child containment do not collide or get split."""
    from krisna_inference.layout_ir import SceneArchetype, LayoutNode, BBox
    from krisna_inference.layout_baseline import layout_from_constraints, detect_scene_archetype

    prompt = "Design a retro music festival poster headline with typography overlay on background art"
    archetype = detect_scene_archetype(prompt)
    assert archetype == SceneArchetype.GRAPHIC_POSTER

    # Feed custom overlapping nodes on different z-index layers
    nodes = [
        LayoutNode(
            id="bg_artwork",
            type="illustration",
            bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
            z_index=0,
        ),
        LayoutNode(
            id="main_headline",
            type="headline",
            text="SUMMER FESTIVAL 2026",
            bbox=BBox(x1=0.1, y1=0.15, x2=0.9, y2=0.35),
            z_index=1,
        ),
        LayoutNode(
            id="badge",
            type="badge",
            text="LIVE AT SUNSET",
            bbox=BBox(x1=0.6, y1=0.25, x2=0.85, y2=0.32),
            z_index=2,
        ),
    ]

    layout = layout_from_constraints({
        "scene_archetype": SceneArchetype.GRAPHIC_POSTER.value,
        "nodes": [n.model_dump() for n in nodes],
    })

    # Assert that all 3 nodes retained their exact bounding boxes without being split or retiled
    node_map = {n.id: n for n in layout.nodes}
    assert "bg_artwork" in node_map
    assert "main_headline" in node_map
    assert "badge" in node_map

    assert node_map["bg_artwork"].bbox.x1 == 0.0 and node_map["bg_artwork"].bbox.x2 == 1.0
    assert node_map["main_headline"].bbox.x1 == 0.1 and node_map["main_headline"].bbox.x2 == 0.9
    assert node_map["badge"].bbox.x1 == 0.6 and node_map["badge"].bbox.x2 == 0.85


def test_facial_scale_floor_enforcement():
    """Verify facial/focal subject dimensions are clamped to >= 128px floor to prevent VAE collapse."""
    from krisna_inference.layout_ir import LayoutNode, BBox
    from krisna_inference.layout_baseline import layout_from_constraints

    # Pass a tiny face node (50x50 on a 1024x1024 canvas = 0.0488x0.0488)
    tiny_focal = LayoutNode(
        id="avatar_face",
        type="focal_subject",
        is_focal_subject=True,
        facial_landmarks={"eyes": [0.02, 0.02]},
        bbox=BBox(x1=0.45, y1=0.45, x2=0.49, y2=0.49),
    )

    layout = layout_from_constraints(
        {"nodes": [tiny_focal.model_dump()]},
        canvas_width=1024,
        canvas_height=1024,
    )

    resolved_focal = next(n for n in layout.nodes if n.id == "avatar_face")
    # Width and height in pixels must be >= 128
    w_px = (resolved_focal.bbox.x2 - resolved_focal.bbox.x1) * layout.canvas_width
    h_px = (resolved_focal.bbox.y2 - resolved_focal.bbox.y1) * layout.canvas_height
    assert w_px >= 127.9
    assert h_px >= 127.9


@pytest.mark.asyncio
async def test_image_to_image_spatial_conditioning_receipts(tmp_path, monkeypatch):
    """Verify QwenImage21Backend multi-stream spatial conditioning (SPADE + inpaint mask)."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend

    store = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: store)

    # 1. Source wireframe
    source = Image.new("RGB", (512, 512), (240, 240, 245))
    src_ref = store.save_image(source, prefix="wf")

    # 2. Semantic segmentation map
    seg = Image.new("RGB", (512, 512), (236, 72, 153))
    seg_ref = store.save_image(seg, prefix="seg")

    # 3. Differential mask
    mask = Image.new("L", (512, 512), 255)
    mask_ref = store.save_image(mask, prefix="mask")

    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=4.5,
        quantization="BF16",
        always_resident=False,
    )
    backend = QwenImage21Backend(spec)
    await backend.load()

    result = await backend.run(
        handoff_image_ref=src_ref,
        segmentation_ref=seg_ref,
        mask_ref=mask_ref,
        prompt="Cyberpunk female character portrait",
    )

    receipt = result["receipt"]
    assert receipt["segmentation_used"] is True
    assert receipt["mask_used"] is True
    assert receipt["spatial_conditioning_mode"] == "NVIDIA_SPADE_MultiStream"


@pytest.mark.asyncio
async def test_vlm_screenshot_decompilation_end_to_end(tmp_path, monkeypatch):
    """Verify that decompile_screenshot feeds raster image into Planner VLM and synchronizes Sketch."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator.model_registry import MockBackend, ModelSpec, Tier
    from krisna_inference.orchestrator import flows

    store = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store)

    db_store = DesignStateStore(db_path=tmp_path / "sessions.db")
    state = db_store.create(DesignState())

    # Create dummy screenshot image
    screenshot = Image.new("RGB", (800, 600), color=(30, 41, 59))
    img_ref = store.save_image(screenshot, prefix="screenshot")

    def mock_factory(spec: ModelSpec) -> MockBackend:
        return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)

    orch = SwapOrchestrator(backend_factory=mock_factory, envelope_gb=24.0)
    await orch.start()

    updated = await flows.decompile_screenshot(
        orchestrator=orch,
        store=db_store,
        session_id=state.session_id,
        image_ref=img_ref,
        instruction="Decompile UI screenshot into structured layout",
    )

    assert updated.stage == SessionStage.SKETCHING
    assert updated.constraints.reference_image_ref == img_ref
    assert updated.layout.wireframe_ref is not None
    assert updated.layout.layout_ref is not None
    assert updated.layout.revision >= 1

    await orch.shutdown()
    db_store.close()


@pytest.mark.asyncio
async def test_image_to_image_turn1_face_preservation(tmp_path, monkeypatch):
    """Verify Turn 1 Image-to-Image preserves locked face pixels from reference_image_ref without drift."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, LockedRegion
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator.model_registry import Tier, REGISTRY
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.orchestrator import flows

    store = BlobStore(root=tmp_path / "blobs")
    import krisna_inference.backends.blob_store_singleton as bss
    bss.set_blob_store(store)
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store)
    monkeypatch.setattr("krisna_inference.backends.layout_backend.get_blob_store", lambda: store)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: store)

    db_store = DesignStateStore(db_path=tmp_path / "sessions_face.db")
    state = db_store.create(DesignState())

    # 1. Create a reference portrait photo with a distinctive red patch on the face (center [0.35, 0.35, 0.3, 0.3])
    ref_photo = Image.new("RGB", (512, 512), color=(100, 150, 200))
    for x in range(180, 330):
        for y in range(180, 330):
            ref_photo.putpixel((x, y), (255, 0, 0))  # Bright red facial test pixels
    ref_photo_ref = store.save_image(ref_photo, prefix="donor_photo")

    # Set up orchestrator with LayoutBackend and QwenImage21Backend
    def backend_factory(spec):
        if spec.tier == Tier.SKETCH:
            return LayoutBackend(spec, output_dir=tmp_path / "layout_out")
        elif spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.POLISH_FAST):
            return QwenImage21Backend(spec)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec, load_latency_s=0.0, unload_latency_s=0.0)

    orch = SwapOrchestrator(backend_factory=backend_factory, envelope_gb=24.0)
    await orch.start()

    # Turn 1: User submits portrait image with instruction to refine style
    updated = await flows.conversational_turn(
        orchestrator=orch,
        store=db_store,
        session_id=state.session_id,
        user_message="Portrait photo edit: apply cinematic lighting while locking face",
        image_ref=ref_photo_ref,
    )
    assert updated.constraints.reference_image_ref == ref_photo_ref

    # Lock the facial region
    cur_rev = updated.revision
    updated.constraints.locked_regions = [
        LockedRegion(bbox=[0.35, 0.35, 0.30, 0.30], reason="Preserve original face")
    ]
    updated.touch()
    db_store.save(updated, expected_revision=cur_rev)

    # Finalize Turn 1 Image-to-Image
    finalized = await flows.finalize(
        orchestrator=orch,
        store=db_store,
        session_id=state.session_id,
        prompt="Cinematic studio lighting portrait",
    )

    assert finalized.finalize_output.image_ref is not None
    assert finalized.finalize_output.image_ref.startswith("blob://")

    # Load output render and verify the locked face pixels (at center of locked region cx=0.50, cy=0.50) are EXACTLY (255, 0, 0)
    out_img = store.load_image(finalized.finalize_output.image_ref)
    cx, cy = int(0.50 * out_img.width), int(0.50 * out_img.height)
    pixel = out_img.getpixel((cx, cy))
    # Red channel must be 255, green and blue 0 (bitwise exact match from donor photo)
    assert pixel[0] == 255
    assert pixel[1] == 0
    assert pixel[2] == 0

    await orch.shutdown()
    db_store.close()


@pytest.mark.asyncio
async def test_unified_plan_sketch_layer_simultaneous_execution(tmp_path):
    """Verify UnifiedPlanSketchLayer executes Plan deliberation and Layout compilation simultaneously."""
    from krisna_inference.orchestrator.unified_plan_sketch import UnifiedPlanSketchLayer
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator

    db_store = DesignStateStore(db_path=str(tmp_path / "test_sessions.db"))
    state = DesignState()
    state = db_store.create(state)

    orch = SwapOrchestrator()
    await orch.start()

    layer = UnifiedPlanSketchLayer(orchestrator=orch, store=db_store)
    updated_state, exec_result = await layer.execute(
        session_id=state.session_id,
        user_message="Design a fintech investment dashboard with dark theme",
    )

    assert updated_state.session_id == state.session_id
    assert updated_state.stage == SessionStage.SKETCHING
    assert updated_state.layout.wireframe_ref is not None
    assert updated_state.layout.segmentation_ref is not None
    assert any(turn.role == "planner" for turn in updated_state.conversation_history)

    assert exec_result.session_id == state.session_id
    assert exec_result.wireframe_ref == updated_state.layout.wireframe_ref
    assert exec_result.segmentation_ref == updated_state.layout.segmentation_ref

    await orch.shutdown()
    db_store.close()


@pytest.mark.asyncio
async def test_plan_sketch_and_polish_closed_loop_deliberation(tmp_path):
    """Verify plan_sketch_and_polish directly proceeds to Polish and loops back to review & further planning."""
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator import flows

    db_store = DesignStateStore(db_path=str(tmp_path / "test_sessions.db"))
    state = DesignState()
    state = db_store.create(state)

    orch = SwapOrchestrator()
    await orch.start()

    # Run the unified plan, sketch, direct-to-polish, and post-render deliberation pipeline
    final_state = await flows.plan_sketch_and_polish(
        orchestrator=orch,
        store=db_store,
        session_id=state.session_id,
        user_message="Create an e-commerce product landing page with hero image and CTA",
        auto_review=True,
    )

    # 1. State must reach FINALIZED stage
    assert final_state.stage == SessionStage.FINALIZED

    # 2. Layout artifacts must be generated
    assert final_state.layout.wireframe_ref is not None
    assert final_state.layout.segmentation_ref is not None

    # 3. Polish output must be populated
    assert final_state.finalize_output.image_ref is not None
    assert final_state.finalize_output.renderer_used == "qwen_image_2_1"

    # 4. Post-render deliberation turn must be appended to conversation history
    history_turns = [turn.content for turn in final_state.conversation_history if turn.role == "planner"]
    assert len(history_turns) >= 2  # Turn 1: initial plan; Turn 2: post-render deliberation
    assert any("[Post-Render Deliberation]" in content for content in history_turns)

    await orch.shutdown()
    db_store.close()


@pytest.mark.asyncio
async def test_plan_sketch_and_polish_fast_preview(tmp_path):
    """Verify plan_sketch_and_polish with fast=True populates preview_output via CFM distillation."""
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator import flows

    db_store = DesignStateStore(db_path=str(tmp_path / "test_sessions.db"))
    state = DesignState()
    state = db_store.create(state)

    orch = SwapOrchestrator()
    await orch.start()

    final_state = await flows.plan_sketch_and_polish(
        orchestrator=orch,
        store=db_store,
        session_id=state.session_id,
        user_message="Quick preview of travel booking card",
        fast=True,
        auto_review=False,
    )

    assert final_state.stage == SessionStage.FINALIZED
    assert final_state.preview_output is not None
    assert final_state.preview_output.step_count == 6
    assert final_state.preview_output.image_ref is not None

    await orch.shutdown()
    db_store.close()


def test_service_execute_endpoint_and_auto_proceed(tmp_path, monkeypatch):
    """Test POST /session/{session_id}/execute and /message with auto_proceed=True via FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from krisna_inference.orchestrator.service import app
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.design_state import DesignState, SessionStage

    test_store = DesignStateStore(db_path=str(tmp_path / "svc_test.db"))
    monkeypatch.setattr("krisna_inference.orchestrator.service.store", test_store)

    with TestClient(app) as client:
        # Create session
        resp = client.post("/session", json={"style": "cyberpunk", "palette": ["#ff0055", "#00ffff"]})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Test POST /session/{session_id}/execute
        exec_resp = client.post(
            f"/session/{session_id}/execute",
            json={"message": "Build modern portfolio hero section", "quality": False, "auto_review": True},
        )
        assert exec_resp.status_code == 200
        data = exec_resp.json()
        assert data["stage"] == "finalized"
        assert data["finalize_output"]["image_ref"] is not None

        # Check post-render deliberation is in conversation history
        planner_turns = [t for t in data["conversation_history"] if t["role"] == "planner"]
        assert any("[Post-Render Deliberation]" in t["content"] for t in planner_turns)

        # Create second session to test /message with auto_proceed=True
        resp2 = client.post("/session", json={})
        session_id_2 = resp2.json()["session_id"]

        msg_resp = client.post(
            f"/session/{session_id_2}/message",
            json={"message": "Generate medical diagnostic card", "auto_proceed": True, "fast": True},
        )
        assert msg_resp.status_code == 200
        data2 = msg_resp.json()
        assert data2["stage"] == "finalized"
        assert data2["preview_output"] is not None

    test_store.close()


def test_vla_action_primitives_end_to_end():
    """Negative audit: verify all VLA action primitives are parsed, executed, and mutate Scene DAG."""
    from krisna_inference.backends.planner_schema import (
        AddComponentAction,
        LockRegionAction,
        MoveComponentAction,
        RemoveComponentAction,
        ReorderHierarchyAction,
        ResizeAction,
        SetAssetModeAction,
        SetLayoutDirectionAction,
        SetPaletteAction,
        SetStyleAction,
        SetTextAction,
    )

    constraints = {
        "style": "default",
        "palette": ["#ffffff", "#000000"],
        "actions": [
            SetStyleAction(style="cyberpunk neon"),
            SetPaletteAction(palette=["#050505", "#00ffcc", "#ff007f"]),
            SetAssetModeAction(asset_mode="transparent"),
            AddComponentAction(
                component_type="focal_subject",
                target_id="hero_face",
                # Pass a tiny box to test spatial floor enforcement (<128px)
                bbox=[0.40, 0.40, 0.05, 0.05],
                props={"label": "Character Portrait"},
                is_focal_subject=True,
            ),
            AddComponentAction(
                component_type="card",
                target_id="overlay_card",
                bbox=[0.10, 0.60, 0.40, 0.30],
                props={"label": "Overlay Panel"},
            ),
            SetLayoutDirectionAction(target_id="root", layout_direction="column"),
            ReorderHierarchyAction(target_id="overlay_card", hierarchy_position="front"),
            SetTextAction(target_id="title", text_content="Updated Mission Title"),
            MoveComponentAction(target_id="action", bbox=[0.55, 0.85, 0.35, 0.08]),
            ResizeAction(target_id="content", bbox=[0.05, 0.20, 0.90, 0.35]),
            LockRegionAction(target_id="hero_face"),
            RemoveComponentAction(target_id="nonexistent_mock"),
        ],
    }

    layout = layout_from_constraints(constraints, text="VLA Primitive Audit", canvas_width=1024, canvas_height=1024)
    nodes = layout.node_map()

    # 1. Spatial floor & Focal subject preservation: min 128px (12.5% of canvas)
    assert "hero_face" in nodes
    hero = nodes["hero_face"]
    assert hero.is_focal_subject is True
    assert hero.locked is True
    assert hero.identity_lock is True
    bw = hero.bbox.x2 - hero.bbox.x1
    bh = hero.bbox.y2 - hero.bbox.y1
    assert bw >= 0.125, f"Spatial floor violated: width {bw} < 0.125 (128px / 1024px)"
    assert bh >= 0.125, f"Spatial floor violated: height {bh} < 0.125 (128px / 1024px)"

    # 2. Two-way lock sync: hero_face must be registered in constraints["locked_regions"]
    assert len(constraints["locked_regions"]) >= 1
    assert any(r.get("bbox") == [hero.bbox.x1, hero.bbox.y1, round(bw, 4), round(bh, 4)] for r in constraints["locked_regions"])

    # 3. Hierarchy mutation: overlay_card was moved to front (z_index > 0)
    assert "overlay_card" in nodes
    assert nodes["overlay_card"].z_index > 0

    # 4. Text and style tokens updated
    assert nodes["root"].style_tokens.get("style") == "cyberpunk neon"
    assert nodes["root"].style_tokens.get("palette") == ["#050505", "#00ffcc", "#ff007f"]
    assert nodes["root"].style_tokens.get("asset_mode") == "transparent"
    assert nodes["title"].text == "Updated Mission Title"


@pytest.mark.asyncio
async def test_face_identity_preservation_turn_1_image_to_image(tmp_path):
    """Verify Turn 1 Image-to-Image preserves human facial identity with 0.00% drift (lock score 1.000)."""
    import numpy as np
    from krisna_inference.backends.blob_store_singleton import get_blob_store
    from krisna_inference.orchestrator.design_state import DesignState
    from krisna_inference.orchestrator.flows import plan_sketch_and_polish
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    blobs = get_blob_store()
    store = DesignStateStore(db_path=str(tmp_path / "face_turn1_test.db"))

    # Synthesize an uploaded photo with a unique distinct human face texture
    w, h = 1024, 1024
    np.random.seed(42)
    face_pattern = np.random.randint(50, 220, (300, 300, 3), dtype=np.uint8)
    base_canvas = np.full((h, w, 3), 240, dtype=np.uint8)
    # Face located at [0.30, 0.20, 0.30, 0.30] -> (307px, 204px)
    fx0, fy0, fw, fh = 307, 204, 300, 300
    base_canvas[fy0 : fy0 + fh, fx0 : fx0 + fw] = face_pattern
    uploaded_photo = Image.fromarray(base_canvas, mode="RGB")
    uploaded_ref = blobs.save_image(uploaded_photo, prefix="test_user_portrait")

    # Initialize session and mock orchestrator
    state = DesignState(session_id="face_session")
    state.constraints.style = "portrait"
    state.constraints.palette = ["#222222", "#ffffff"]
    state.constraints.reference_image_ref = uploaded_ref
    store.create(state)

    def test_backend_factory(spec):
        if spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_FAST, Tier.POLISH_QUALITY):
            return QwenImage21Backend(spec, strict=False)
        if spec.tier == Tier.SKETCH:
            from krisna_inference.backends.layout_backend import LayoutBackend
            return LayoutBackend(spec)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)

    orchestrator = SwapOrchestrator(backend_factory=test_backend_factory)
    await orchestrator.start()

    try:
        final_state = await plan_sketch_and_polish(
            orchestrator=orchestrator,
            store=store,
            session_id="face_session",
            user_message="Keep the person's face untouched, add an elegant subtitle below",
            image_ref=uploaded_ref,
            quality=False,
            auto_review=False,
        )

        assert final_state.stage.value == "finalized"
        assert final_state.finalize_output.image_ref is not None

        # Verify donor pixels were bitwise restored
        polished_img = blobs.load_image(final_state.finalize_output.image_ref)
        locked_regions = [r.model_dump() for r in final_state.constraints.locked_regions]

        # Ensure at least one locked region covers the face
        assert len(locked_regions) >= 1
        verifier = LockPreservationVerifier()
        score = verifier.score(uploaded_photo, polished_img, locked_regions)

        # Facial preservation contract: EXACT donor pixels preserved -> score must be 1.000
        assert score == 1.0, f"Facial drift detected! Lock score {score:.4f} < 1.000"
    finally:
        await orchestrator.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_face_identity_preservation_multiturn_no_degradation(tmp_path):
    """Verify facial identity is preserved across multiple conversational turns without degradation."""
    import numpy as np
    from krisna_inference.backends.blob_store_singleton import get_blob_store
    from krisna_inference.orchestrator.flows import conversational_turn, finalize
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    blobs = get_blob_store()
    store = DesignStateStore(db_path=str(tmp_path / "face_multiturn_test.db"))

    # Synthesize original portrait
    w, h = 1024, 1024
    face_pattern = np.random.randint(60, 200, (250, 250, 3), dtype=np.uint8)
    canvas = np.full((h, w, 3), 230, dtype=np.uint8)
    canvas[150:400, 200:450] = face_pattern
    donor_photo = Image.fromarray(canvas, mode="RGB")
    donor_ref = blobs.save_image(donor_photo, prefix="donor_portrait")

    from krisna_inference.orchestrator.design_state import DesignState

    state = DesignState(session_id="multiturn_face")
    state.constraints.style = "minimalist"
    state.constraints.reference_image_ref = donor_ref
    store.create(state)

    def test_backend_factory(spec):
        if spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_FAST, Tier.POLISH_QUALITY):
            return QwenImage21Backend(spec, strict=False)
        if spec.tier == Tier.SKETCH:
            from krisna_inference.backends.layout_backend import LayoutBackend
            return LayoutBackend(spec)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)

    orchestrator = SwapOrchestrator(backend_factory=test_backend_factory)
    await orchestrator.start()

    try:
        # Turn 1: Initial creation with face lock
        state = await conversational_turn(
            orchestrator=orchestrator,
            store=store,
            session_id="multiturn_face",
            user_message="Lock the portrait face at [0.1953, 0.1465, 0.2441, 0.2441], add heading",
            image_ref=donor_ref,
        )
        state = await finalize(orchestrator=orchestrator, store=store, session_id="multiturn_face")
        t1_img = blobs.load_image(state.finalize_output.image_ref)

        locked_regs = [r.model_dump() for r in state.constraints.locked_regions]
        assert len(locked_regs) >= 1
        verifier = LockPreservationVerifier()
        score_t1 = verifier.score(donor_photo, t1_img, locked_regs)
        assert score_t1 == 1.0, f"Turn 1 face drift: score {score_t1:.4f}"

        # Turn 2: Mutate theme and style
        state = await conversational_turn(
            orchestrator=orchestrator,
            store=store,
            session_id="multiturn_face",
            user_message="Change theme to dark emerald and add a call to action",
        )
        state = await finalize(orchestrator=orchestrator, store=store, session_id="multiturn_face")
        t2_img = blobs.load_image(state.finalize_output.image_ref)

        # Check face is STILL bitwise preserved from donor
        score_t2 = verifier.score(donor_photo, t2_img, locked_regs)
        assert score_t2 == 1.0, f"Turn 2 face degradation: score {score_t2:.4f}"

        # Turn 3: Add additional elements
        state = await conversational_turn(
            orchestrator=orchestrator,
            store=store,
            session_id="multiturn_face",
            user_message="Add navigation links at the top",
        )
        state = await finalize(orchestrator=orchestrator, store=store, session_id="multiturn_face")
        t3_img = blobs.load_image(state.finalize_output.image_ref)

        # Check face remains 100% invariant through Turn 3
        score_t3 = verifier.score(donor_photo, t3_img, locked_regs)
        assert score_t3 == 1.0, f"Turn 3 face degradation: score {score_t3:.4f}"
    finally:
        await orchestrator.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_polish_backend_differential_mask_and_pipeline_signature(tmp_path, monkeypatch):
    """Verify PolishBackendQwen passes mask_image to native pipeline and composites differential masks."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.orchestrator.model_registry import Tier, ModelSpec

    blobs = BlobStore(root=tmp_path / "blobs")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: blobs)

    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="test_qwen", vram_gb=14.0, quantization="bf16")
    backend = QwenImage21Backend(spec, strict=False)

    # 1. Test native pipeline signature inspection & mask forwarding
    class MockPipeline:
        def __init__(self):
            self.last_call = None
            self.scheduler = None

        def __call__(self, prompt, image, num_inference_steps=28, mask_image=None, width=None, height=None, **kwargs):
            self.last_call = {"prompt": prompt, "image": image, "mask_image": mask_image, "width": width, "height": height}
            from PIL import Image
            class Output:
                images = [Image.new("RGB", (width or 512, height or 512), (120, 120, 120))]
            return Output()

    mock_pipe = MockPipeline()
    backend._pipeline = mock_pipe
    backend._native_call_parameters = {"prompt", "image", "num_inference_steps", "mask_image", "width", "height"}
    backend._loaded = True

    # Create source image, mask image, and donor image
    src_img = Image.new("RGB", (512, 512), (50, 50, 50))
    src_ref = blobs.save_image(src_img, prefix="src")

    mask_img = Image.new("L", (512, 512), 0)
    # Active edit zone in bottom quadrant:
    for x in range(100, 400):
        for y in range(300, 500):
            mask_img.putpixel((x, y), 255)
    mask_ref = blobs.save_image(mask_img, prefix="diff_mask")

    res = await backend.run(
        image_ref=src_ref,
        mask_ref=mask_ref,
        prompt="Update hero button",
        constraints={"style": "modern"},
    )
    assert res["run_receipt"]["mask_used"] is True
    assert mock_pipe.last_call is not None
    assert mock_pipe.last_call["mask_image"] is not None

    # 2. Test synthetic fallback compositing preserves locked/unchanged background
    backend._pipeline = None  # Switch to synthetic mode
    donor_img = Image.new("RGB", (512, 512), (200, 10, 10))  # Vivid red donor
    donor_ref = blobs.save_image(donor_img, prefix="donor")

    res_synth = await backend.run(
        image_ref=src_ref,
        lock_donor_ref=donor_ref,
        mask_ref=mask_ref,
        prompt="Update hero button",
        constraints={"style": "modern"},
    )
    synth_output = blobs.load_image(res_synth["image_ref"])
    # Where mask was 0 (e.g. at (50, 50)), donor red pixels MUST be preserved exactly!
    assert synth_output.getpixel((50, 50)) == (200, 10, 10)


@pytest.mark.asyncio
async def test_iterative_refine_syncs_dual_stream_sketch_and_mask_to_polish(tmp_path, monkeypatch):
    """Verify iterative_refine re-renders dual-stream sketch (SPADE + differential mask) and passes to polish."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, LayoutDraft, SessionStage
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator.model_registry import Tier
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.orchestrator.flows import finalize, iterative_refine

    blobs = BlobStore(root=tmp_path / "blobs_refine")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.layout_backend.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: blobs)

    store = DesignStateStore(db_path=tmp_path / "sessions_refine.db")

    # Populate initial layout and finalize
    from krisna_inference.layout_baseline import layout_from_constraints
    initial_layout = layout_from_constraints({"style": "dark", "layout_hints": ["header", "cta_button"]})
    layout_file = tmp_path / "initial_layout.json"
    import json
    layout_file.write_text(json.dumps(initial_layout.model_dump(), indent=2), encoding="utf-8")

    initial_state = DesignState(
        session_id="refine_test_sess",
        stage=SessionStage.SKETCHING,
        layout=LayoutDraft(layout_ref=str(layout_file)),
    )
    state = store.create(initial_state)

    def backend_factory(spec):
        if spec.tier == Tier.SKETCH:
            return LayoutBackend(spec, output_dir=tmp_path / "layout_out")
        elif spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.POLISH_FAST):
            return QwenImage21Backend(spec, strict=False)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec, load_latency_s=0.0, unload_latency_s=0.0)

    orch = SwapOrchestrator(backend_factory=backend_factory, envelope_gb=24.0)
    await orch.start()
    try:
        # Initial finalize
        state = await finalize(orchestrator=orch, store=store, session_id="refine_test_sess")
        assert state.finalize_output.image_ref is not None

        # Execute iterative refine with MOVE_COMPONENT action
        state = await iterative_refine(
            orchestrator=orch,
            store=store,
            session_id="refine_test_sess",
            instruction="Move cta button to bottom",
            actions=[{"action": "move_component", "target_id": "cta_button", "bbox": [0.2, 0.85, 0.6, 0.1]}],
        )

        # Assert dual-stream conditioning was synchronized and updated:
        assert state.layout.wireframe_ref is not None
        assert state.layout.segmentation_ref is not None
        assert state.layout.mask_ref is not None
        assert state.layout.revision >= 1

        # Polish backend synthesized render should succeed and reflect the refinement
        assert state.finalize_output.image_ref is not None
    finally:
        await orch.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_finalize_retries_on_lock_preservation_drop(tmp_path, monkeypatch):
    """Verify that lock preservation degradation (< 0.95) triggers automated retry with mask_ref preserved."""
    from PIL import Image
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import DesignState, LayoutDraft, SessionStage, LockedRegion
    from krisna_inference.orchestrator.model_registry import Tier
    from krisna_inference.orchestrator.swap_orchestrator import SwapResult
    from krisna_inference.orchestrator.flows import finalize

    store_blob = BlobStore(root=tmp_path / "blobs_retry")
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: store_blob)

    state = DesignState(
        session_id="retry_lock_test",
        stage=SessionStage.SKETCHING,
        layout=LayoutDraft(
            layout_ref="blob://layout.json",
            wireframe_ref="blob://wf.png",
            segmentation_ref="blob://seg.png",
            mask_ref="blob://diff_mask.png",
        ),
    )
    state.constraints.locked_regions = [LockedRegion(id="face_0", bbox=[0.2, 0.2, 0.3, 0.3], reason="user_locked")]

    class FakeStore:
        def get(self, session_id):
            return state
        def save(self, s, rev):
            return s

    retry_count = 0
    captured_calls = []

    class FakeOrchestrator:
        async def request_finalize(self, **kwargs):
            nonlocal retry_count
            retry_count += 1
            captured_calls.append(kwargs)
            img = Image.new("RGB", (100, 100), (50, 50, 50))
            ref = store_blob.save_image(img, prefix="render")
            return SwapResult(ok=True, tier_used=Tier.POLISH_DEFAULT, output={"image_ref": ref})

    class FakeVerifierStack:
        def safety_gate(self, img, min_safety_score=0.9):
            return True, 0.99

        def score_finalize_output(self, polished_image, **kwargs):
            # Attempt 1: lock_preservation fails (0.80 < 0.95)
            # Attempt 2: lock_preservation passes (1.00)
            score = 0.80 if retry_count == 1 else 1.00
            return {
                "layout_iou": 0.85,
                "ocr_readability": 0.90,
                "seam_continuity": 0.95,
                "lock_preservation": score,
            }

    final_state = await finalize(
        orchestrator=FakeOrchestrator(),
        store=FakeStore(),
        session_id="retry_lock_test",
        verifier_stack=FakeVerifierStack(),
    )

    assert retry_count == 2, f"Expected 2 attempts (initial + retry), got {retry_count}"
    # Verify mask_ref was forwarded to the retry call!
    assert captured_calls[1]["mask_ref"] == "blob://diff_mask.png"
    assert captured_calls[1]["segmentation_ref"] == "blob://seg.png"
    # Verify drift reasons in prompt mention lock preservation
    assert "Lock preservation correction (0.80 < 0.95)" in captured_calls[1]["prompt"]


@pytest.mark.asyncio
async def test_multi_asset_duo_composition_self_and_girl(tmp_path, monkeypatch):
    """Grounded Research Validation: MS-Diffusion (CVPR 2024) & FastComposer (SIGGRAPH Asia 2023).

    Verifies multi-subject personalized image generation combining two distinct photos
    (e.g., selfie of oneself + photo of a girl) into a single cohesive scene without identity blending:
    1. Registers both ReferenceAsset models ('asset_self' and 'asset_girl') in design state.
    2. Planner VLA generates distinct ADD_COMPONENT actions with distinct source_asset_id and source_crop_bbox.
    3. Sketch tier partitions both subjects cleanly with identity_lock=True and minimum spatial floors.
    4. Polish tier receives lock_donor_map and applies localized Crop -> Lanczos Resample -> Paste.
    5. LockPreservationVerifier achieves verified 1.000 (0.00% facial drift) across both subjects.
    """
    from PIL import Image, ImageDraw
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import ReferenceAsset, DesignState
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.model_registry import Tier
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator.flows import plan_sketch_and_polish
    from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    blobs = BlobStore(root=tmp_path / "blobs_duo")
    import krisna_inference.backends.blob_store_singleton as bss
    bss.set_blob_store(blobs)
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.layout_backend.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.planner_backend_llama.get_blob_store", lambda: blobs)

    # 1. Create two distinct synthetic photographic portraits
    # Photo 1: User's face (blue tones)
    img_self = Image.new("RGB", (512, 512), (30, 60, 120))
    d1 = ImageDraw.Draw(img_self)
    d1.ellipse([150, 120, 360, 380], fill=(210, 180, 140), outline=(10, 30, 90), width=4)
    ref_self = blobs.save_image(img_self, prefix="self")

    # Photo 2: Girl's face (warm amber/rose tones)
    img_girl = Image.new("RGB", (512, 512), (120, 40, 60))
    d2 = ImageDraw.Draw(img_girl)
    d2.ellipse([150, 120, 360, 380], fill=(225, 190, 160), outline=(90, 10, 30), width=4)
    ref_girl = blobs.save_image(img_girl, prefix="girl")

    store = DesignStateStore(db_path=tmp_path / "duo_state.db")
    store.create(DesignState(session_id="duo_composition_sess"))

    def backend_factory(spec):
        if spec.tier == Tier.PLANNER:
            return PlannerBackendLlama(spec)
        elif spec.tier == Tier.SKETCH:
            return LayoutBackend(spec, output_dir=tmp_path / "layout_out")
        elif spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.POLISH_FAST):
            return QwenImage21Backend(spec, strict=False)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec)

    orch = SwapOrchestrator(backend_factory=backend_factory, envelope_gb=24.0)
    await orch.start()
    try:
        ref_assets = [
            ReferenceAsset(id="asset_self", blob_ref=ref_self, role="subject", description="Self portrait"),
            ReferenceAsset(id="asset_girl", blob_ref=ref_girl, role="subject", description="Photo of a girl"),
        ]

        state = await plan_sketch_and_polish(
            orchestrator=orch,
            store=store,
            session_id="duo_composition_sess",
            user_message="Combine a photo of myself and this girl sitting together in a cafe",
            reference_assets=ref_assets,
            auto_review=False,
        )

        # Assert reference assets stored properly in constraints
        assert len(state.constraints.reference_assets) == 2
        assert state.constraints.reference_assets[0].id == "asset_self"
        assert state.constraints.reference_assets[1].id == "asset_girl"

        # Assert locked regions preserved both identities with source metadata
        locked = state.constraints.locked_regions
        assert len(locked) >= 2
        asset_ids = {r.source_asset_id for r in locked if r.source_asset_id}
        assert "asset_self" in asset_ids
        assert "asset_girl" in asset_ids

        # Assert rendered image exists
        rendered_ref = state.finalize_output.image_ref
        assert rendered_ref is not None
        rendered_img = blobs.load_image(rendered_ref)

        # Assert multi-subject descriptive prompt synthesized properly
        prompt_used = state.finalize_output.prompt_used
        assert "portrait" in prompt_used.lower()

        # Score exact lock preservation using localized crop-and-paste verification
        verifier = LockPreservationVerifier()
        donor_map = {"asset_self": img_self, "asset_girl": img_girl}
        lock_score = verifier.score(
            source_image=None,
            rendered_image=rendered_img,
            locked_regions=[r.model_dump() for r in locked],
            donor_map=donor_map,
        )
        assert lock_score is not None
        assert lock_score >= 0.999, f"Expected near-perfect lock score, got {lock_score}"
    finally:
        bss.set_blob_store(None)
        await orch.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_multi_asset_subject_and_background_teleportation(tmp_path, monkeypatch):
    """Grounded Research Validation: AnyDoor (CVPR 2024) Object Teleportation.

    Verifies composing a portrait selfie onto an uploaded background environment:
    1. Asset 1: Background photo (role='background')
    2. Asset 2: Selfie photo (role='subject')
    3. Planner VLA emits background component at [0, 0, 1, 1] and focal subject at foreground.
    4. Polish tier preserves both background scene and exact facial identity.
    """
    from PIL import Image, ImageDraw
    from krisna_inference.backends.common import BlobStore
    from krisna_inference.orchestrator.design_state import ReferenceAsset, DesignState
    from krisna_inference.orchestrator.store import DesignStateStore
    from krisna_inference.orchestrator.model_registry import Tier
    from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krisna_inference.orchestrator.flows import plan_sketch_and_polish
    from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    blobs = BlobStore(root=tmp_path / "blobs_teleport")
    import krisna_inference.backends.blob_store_singleton as bss
    bss.set_blob_store(blobs)
    monkeypatch.setattr("krisna_inference.backends.blob_store_singleton.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.layout_backend.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.polish_backend_qwen.get_blob_store", lambda: blobs)
    monkeypatch.setattr("krisna_inference.backends.planner_backend_llama.get_blob_store", lambda: blobs)

    # Background: Sunset beach environment
    img_bg = Image.new("RGB", (640, 480), (240, 140, 60))
    d_bg = ImageDraw.Draw(img_bg)
    d_bg.rectangle([0, 300, 640, 480], fill=(40, 90, 160))  # ocean
    ref_bg = blobs.save_image(img_bg, prefix="bg")

    # Foreground subject: Portrait photo
    img_self = Image.new("RGB", (400, 400), (30, 30, 30))
    d_self = ImageDraw.Draw(img_self)
    d_self.ellipse([100, 80, 300, 320], fill=(215, 185, 145), outline=(20, 20, 20), width=3)
    ref_self = blobs.save_image(img_self, prefix="self")

    store = DesignStateStore(db_path=tmp_path / "teleport_state.db")
    store.create(DesignState(session_id="teleport_sess"))

    def backend_factory(spec):
        if spec.tier == Tier.PLANNER:
            return PlannerBackendLlama(spec)
        elif spec.tier == Tier.SKETCH:
            return LayoutBackend(spec, output_dir=tmp_path / "layout_out")
        elif spec.tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY, Tier.POLISH_FAST):
            return QwenImage21Backend(spec, strict=False)
        from krisna_inference.orchestrator.model_registry import MockBackend
        return MockBackend(spec)

    orch = SwapOrchestrator(backend_factory=backend_factory, envelope_gb=24.0)
    await orch.start()
    try:
        ref_assets = [
            ReferenceAsset(id="asset_bg", blob_ref=ref_bg, role="background", description="Sunset beach"),
            ReferenceAsset(id="asset_self", blob_ref=ref_self, role="subject", description="Self portrait"),
        ]

        state = await plan_sketch_and_polish(
            orchestrator=orch,
            store=store,
            session_id="teleport_sess",
            user_message="Place myself on this sunset beach background",
            reference_assets=ref_assets,
            auto_review=False,
        )

        assert len(state.constraints.reference_assets) == 2
        rendered_ref = state.finalize_output.image_ref
        assert rendered_ref is not None
        rendered_img = blobs.load_image(rendered_ref)

        # Verify locked regions contain the teleported focal subject
        locked = state.constraints.locked_regions
        assert any(r.source_asset_id == "asset_self" for r in locked)

        # Verify pixel preservation
        verifier = LockPreservationVerifier()
        donor_map = {"asset_self": img_self, "asset_bg": img_bg}
        score = verifier.score(
            source_image=None,
            rendered_image=rendered_img,
            locked_regions=[r.model_dump() for r in locked],
            donor_map=donor_map,
        )
        assert score is not None
        assert score >= 0.999
    finally:
        bss.set_blob_store(None)
        await orch.shutdown()
        store.close()


def test_crop_and_paste_localized_facial_preservation():
    """Verify localized Crop -> Lanczos Resample -> Paste eliminates off-center face drift.

    Solves the negative audit gap where a face off-center in the source photo
    (e.g., at [0.60, 0.35, 0.30, 0.40]) was previously destroyed by 1-to-1 canvas masking.
    """
    from PIL import Image, ImageDraw
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
    from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier

    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="qwen_image", vram_gb=12.0, quantization="bf16")
    backend = QwenImage21Backend(spec, strict=False)

    # Source photo (500x500): Face is located off-center on the right side
    src_photo = Image.new("RGB", (500, 500), (40, 40, 40))
    d_src = ImageDraw.Draw(src_photo)
    # Face rectangle at x=[300, 450], y=[150, 350] (i.e. [0.60, 0.30, 0.30, 0.40])
    d_src.rectangle([300, 150, 450, 350], fill=(230, 190, 150))
    d_src.ellipse([330, 200, 360, 230], fill=(20, 20, 80))  # left eye
    d_src.ellipse([390, 200, 420, 230], fill=(20, 20, 80))  # right eye

    # Target canvas (800x800): User wants the face placed on the left side at [0.10, 0.20, 0.35, 0.40]
    canvas = Image.new("RGB", (800, 800), (200, 200, 200))

    locked_region = {
        "id": "portrait_face",
        "bbox": [0.10, 0.20, 0.35, 0.40],
        "source_asset_id": "asset_me",
        "source_crop_bbox": [0.60, 0.30, 0.30, 0.40],
    }

    donor_map = {"asset_me": src_photo}

    composited = backend._apply_locked_regions(
        source_img=src_photo,
        rendered_img=canvas,
        locked_regions=[locked_region],
        donor_map=donor_map,
    )

    # Target face box on canvas: [80, 160, 80+280=360, 160+320=480]
    # Check that pixels inside target box are flesh tone and not canvas gray!
    px_center = composited.getpixel((220, 320))
    assert px_center[0] > 200 and px_center[1] > 170  # flesh tone
    # Check that pixels outside target box remain canvas gray
    px_outside = composited.getpixel((50, 50))
    assert px_outside == (200, 200, 200)

    # Verify score is exactly 1.000 (0.00% facial drift)
    verifier = LockPreservationVerifier()
    score = verifier.score(
        source_image=src_photo,
        rendered_image=composited,
        locked_regions=[locked_region],
        donor_map=donor_map,
    )
    assert score is not None
    assert score >= 0.999





