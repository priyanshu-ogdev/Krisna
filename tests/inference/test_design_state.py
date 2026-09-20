from __future__ import annotations

from krisna_inference.orchestrator.design_state import DesignState, SessionStage


def test_default_state_shape_matches_prd_5_1():
    state = DesignState()
    wire = state.to_wire()
    assert set(wire.keys()) == {
        "session_id",
        "stage",
        "conversation_history",
        "constraints",
        "sketch_tokens",
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
    state.sketch_tokens.vq_tokens = "grid_ref_1"
    assert state.is_finalize_eligible()


def test_finalize_eligibility_false_once_finalized():
    state = DesignState()
    state.sketch_tokens.vq_tokens = "grid_ref_1"
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
    from krisna_inference.orchestrator.design_state import LockedRegion

    state = DesignState()
    state.constraints.locked_regions.append(
        LockedRegion(bbox=(0.1, 0.1, 0.3, 0.2), reason="logo must not move")
    )
    wire = state.to_wire()
    assert wire["constraints"]["locked_regions"][0]["reason"] == "logo must not move"
