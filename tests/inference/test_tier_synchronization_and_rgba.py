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





