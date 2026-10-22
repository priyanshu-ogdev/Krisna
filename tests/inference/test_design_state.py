from __future__ import annotations

from krivex_inference.orchestrator.design_state import DesignState, SessionStage


def test_default_state_shape_matches_prd_5_1():
    state = DesignState()
    wire = state.to_wire()
    assert set(wire.keys()) == {
        "session_id",
        "stage",
        "conversation_history",
        "constraints",
        "layout",
        "finalize_output",
        "critique",
        "preference_pair_refs",
    }
    assert wire["stage"] == "conversing"


def test_append_turn_bumps_revision():
    state = DesignState()
    rev0 = state.revision
    state.append_turn("user", "make it blue")
    assert state.revision == rev0 + 1
    assert state.conversation_history[-1].role == "user"
    assert state.conversation_history[-1].content == "make it blue"


def test_finalize_eligibility_requires_sketch_tokens():
    state = DesignState()
    assert not state.is_finalize_eligible()
    state.layout.layout_ref = "layout://test/1.json"
    assert state.is_finalize_eligible()


def test_finalize_eligibility_false_once_finalized():
    state = DesignState()
    state.layout.layout_ref = "layout://test/1.json"
    state.stage = SessionStage.FINALIZED
    assert not state.is_finalize_eligible()


def test_critique_eligibility_requires_finalized_render():
    state = DesignState()
    assert not state.is_critique_eligible()
    state.stage = SessionStage.FINALIZED
    assert not state.is_critique_eligible()  # no image_ref yet
    state.finalize_output.image_ref = "render://x/final.png"
    assert state.is_critique_eligible()


def test_locked_regions_roundtrip():
    from krivex_inference.orchestrator.design_state import LockedRegion

    state = DesignState()
    state.constraints.locked_regions.append(
        LockedRegion(bbox=(0.1, 0.1, 0.3, 0.2), reason="logo must not move")
    )
    wire = state.to_wire()
    assert wire["constraints"]["locked_regions"][0]["reason"] == "logo must not move"


def test_refine_eligibility():
    state = DesignState()
    assert not state.is_refine_eligible()
    state.stage = SessionStage.FINALIZED
    assert not state.is_refine_eligible()  # no render yet
    state.finalize_output.image_ref = "render://x/final.png"
    assert state.is_refine_eligible()
    state.stage = SessionStage.SKETCHING
    assert not state.is_refine_eligible()


def test_instructlayout_formatting():
    from krivex_inference.backends.planner_backend_llama import PlannerBackendLlama

    layout_tree = {
        "nodes": [
            {"id": "nav_1", "type": "navbar", "bbox": [0.0, 0.0, 1.0, 0.08], "locked": True},
            {"id": "hero_cta", "type": "button", "bbox": [0.4, 0.6, 0.2, 0.08], "text": "Get Started"},
        ]
    }
    prefix = PlannerBackendLlama._format_layout_graph_prefix(layout_tree)
    assert "[Structured Layout IR Graph (InstructLayout/TPAMI 2025)]:" in prefix
    assert "id=nav_1, type=navbar" in prefix
    assert "locked=True" in prefix
    assert "text='Get Started'" in prefix
