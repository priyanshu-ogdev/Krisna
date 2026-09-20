from __future__ import annotations

import pytest

from krisna_inference.orchestrator import flows
from krisna_inference.orchestrator.design_state import SessionStage
from krisna_inference.orchestrator.model_registry import Tier
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator


@pytest.mark.asyncio
async def test_conversational_turn_flow(started_orchestrator: SwapOrchestrator, store: DesignStateStore):
    state = store.create()
    result = await flows.conversational_turn(started_orchestrator, store, state.session_id, "make a login screen")
    assert result.stage == SessionStage.SKETCHING
    assert len(result.conversation_history) == 2  # user + planner
    assert result.conversation_history[0].role == "user"
    assert result.sketch_tokens.revision == 1
    # D2: First turn captures original_intent
    assert result.constraints.original_intent == "make a login screen"

    # Second turn preserves original_intent without overwriting
    result2 = await flows.conversational_turn(started_orchestrator, store, state.session_id, "change buttons to blue")
    assert result2.constraints.original_intent == "make a login screen"


@pytest.mark.asyncio
async def test_finalize_flow_requires_sketch(started_orchestrator: SwapOrchestrator, store: DesignStateStore):
    state = store.create()  # no sketch tokens yet
    with pytest.raises(flows.FlowError):
        await flows.finalize(started_orchestrator, store, state.session_id)


@pytest.mark.asyncio
async def test_finalize_flow_happy_path(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    result = await flows.finalize(started_orchestrator, store, sketch_ready_state.session_id)
    assert result.stage == SessionStage.FINALIZED
    assert result.finalize_output.renderer_used == "z_image_turbo"
    assert result.finalize_output.image_ref is not None
    # Baseline restored after the flow completes.
    assert started_orchestrator.conversation_available()


def _patch_blob_store(monkeypatch, tmp_path, subdir: str):
    """Point the blob store singleton at an isolated tmp dir and return a
    real blob:// ref to a tiny in-memory image — needed because
    MockBackend's own image_ref (`render://mock/...`) never starts with
    `blob://`, so it never exercises the verifier/safety-gate branch in
    flows.finalize. A real blob ref is required to reach that branch at
    all."""
    import krisna_inference.backends.blob_store_singleton as blob_singleton
    from PIL import Image

    monkeypatch.setenv("KRISNA_BLOB_ROOT", str(tmp_path / subdir))
    monkeypatch.setattr(blob_singleton, "_instance", None)
    return blob_singleton.get_blob_store().save_image(Image.new("RGB", (8, 8)), prefix="test")


class TestFinalizeFlowSafetyGate:
    """Regression coverage for the safety-gate wiring fix — VerifierStack.
    safety_gate() existed but was never called anywhere before this fix
    (see docs/review/14_safety_gate_wiring.md). These tests exercise the
    actual finalize() code path with a real blob:// image ref, not just
    the gate method in isolation."""

    @pytest.mark.asyncio
    async def test_unsafe_image_blocks_finalize_and_rolls_back_stage(
        self, started_orchestrator, store, sketch_ready_state, tmp_path, monkeypatch
    ):
        blob_ref = _patch_blob_store(monkeypatch, tmp_path, "blobs_unsafe")

        async def fake_run(**kwargs):
            return {"tier": "polish_default", "image_ref": blob_ref, "verifier_scores": {}}

        started_orchestrator.backends[Tier.POLISH_DEFAULT].run = fake_run

        class FakeVerifierStack:
            def safety_gate(self, image, min_safety_score=0.9):
                return False, 0.1  # unsafe

            def score_finalize_output(self, *a, **k):
                raise AssertionError(
                    "score_finalize_output must not run once the safety gate has failed"
                )

        with pytest.raises(flows.FlowError, match="safety gate failed"):
            await flows.finalize(
                started_orchestrator, store, sketch_ready_state.session_id,
                verifier_stack=FakeVerifierStack(),
            )

        reloaded = store.get(sketch_ready_state.session_id)
        assert reloaded.stage == SessionStage.SKETCHING  # rolled back, not FINALIZED
        assert reloaded.finalize_output.image_ref != blob_ref or reloaded.stage != SessionStage.FINALIZED

    @pytest.mark.asyncio
    async def test_safe_image_passes_gate_and_scores_still_populate(
        self, started_orchestrator, store, sketch_ready_state, tmp_path, monkeypatch
    ):
        blob_ref = _patch_blob_store(monkeypatch, tmp_path, "blobs_safe")

        async def fake_run(**kwargs):
            return {"tier": "polish_default", "image_ref": blob_ref, "verifier_scores": {}}

        started_orchestrator.backends[Tier.POLISH_DEFAULT].run = fake_run

        class FakeVerifierStack:
            def safety_gate(self, image, min_safety_score=0.9):
                return True, 0.99

            def score_finalize_output(self, *a, **k):
                return {
                    "clip_alignment": 0.5, "ocr_readability": None,
                    "layout_iou": 0.5, "aesthetic": 0.5, "handoff_consistency": None,
                }

        result = await flows.finalize(
            started_orchestrator, store, sketch_ready_state.session_id,
            verifier_stack=FakeVerifierStack(),
        )
        assert result.stage == SessionStage.FINALIZED
        assert result.finalize_output.image_ref == blob_ref
        assert result.finalize_output.verifier_scores.clip_alignment == 0.5


@pytest.mark.asyncio
async def test_finalize_flow_quality_tier(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    result = await flows.finalize(started_orchestrator, store, sketch_ready_state.session_id, quality=True)
    assert result.finalize_output.renderer_used == "qwen_image_edit_2511"


@pytest.mark.asyncio
async def test_finalize_flow_handoff_hook_invoked(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    calls = []

    def hook(vq_tokens):
        calls.append(vq_tokens)
        return f"pixel_ref_for::{vq_tokens}"

    await flows.finalize(
        started_orchestrator, store, sketch_ready_state.session_id, handoff_hook=hook
    )
    assert calls == ["vq_grid_ref_abc123"]


@pytest.mark.asyncio
async def test_finalize_flow_oom_rolls_back_stage(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    started_orchestrator.backends[Tier.POLISH_DEFAULT].fail_loads = 999
    with pytest.raises(flows.FlowError):
        await flows.finalize(started_orchestrator, store, sketch_ready_state.session_id)

    reloaded = store.get(sketch_ready_state.session_id)
    # Rolled back to SKETCHING, not stuck in FINALIZING.
    assert reloaded.stage == SessionStage.SKETCHING
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_critique_flow_requires_finalized_render(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    with pytest.raises(flows.FlowError):
        await flows.critique_pass(started_orchestrator, store, sketch_ready_state.session_id)


@pytest.mark.asyncio
async def test_critique_flow_happy_path(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, finalized_state
):
    result = await flows.critique_pass(started_orchestrator, store, finalized_state.session_id)
    assert result.stage == SessionStage.FINALIZED  # returns to FINALIZED, not stuck CRITIQUING
    assert result.critique.requested is True
    assert result.critique.source == "gemma4_31b"
    assert result.critique.result is not None
    assert result.critique.result.critique_source == "gemma4_31b_frozen"


@pytest.mark.asyncio
async def test_critique_flow_oom_rolls_back_stage(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, finalized_state
):
    started_orchestrator.backends[Tier.CRITIC].fail_loads = 999
    with pytest.raises(flows.FlowError):
        await flows.critique_pass(started_orchestrator, store, finalized_state.session_id)

    reloaded = store.get(finalized_state.session_id)
    assert reloaded.stage == SessionStage.FINALIZED
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_critique_flow_builds_preference_pair_when_given_comparison(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, finalized_state
):
    from krisna_training.dpo.preference_store import PreferenceStore

    pref_store = PreferenceStore(db_path=":memory:")
    try:
        result = await flows.critique_pass(
            started_orchestrator,
            store,
            finalized_state.session_id,
            preference_store=pref_store,
            compare_against={"image_ref": "blob://previous_candidate.png", "score": 0.2},
        )
        assert len(result.preference_pair_refs) == 1
        pairs = pref_store.list(source="gemma_critique")
        assert len(pairs) == 1
        assert pairs[0].session_id == finalized_state.session_id
    finally:
        pref_store.close()


@pytest.mark.asyncio
async def test_critique_flow_no_pair_without_preference_store(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, finalized_state
):
    result = await flows.critique_pass(started_orchestrator, store, finalized_state.session_id)
    assert result.preference_pair_refs == []


@pytest.mark.asyncio
async def test_full_session_lifecycle(started_orchestrator: SwapOrchestrator, store: DesignStateStore):
    """Smoke test running all three flows back to back on one session,
    mirroring the product use case in PRD §4."""
    state = store.create()
    state = await flows.conversational_turn(started_orchestrator, store, state.session_id, "poster for a coffee shop")
    assert state.stage == SessionStage.SKETCHING

    state = await flows.finalize(started_orchestrator, store, state.session_id)
    assert state.stage == SessionStage.FINALIZED

    state = await flows.critique_pass(started_orchestrator, store, state.session_id)
    assert state.stage == SessionStage.FINALIZED
    assert state.critique.result is not None

    # System should be conversation-available at the very end.
    assert started_orchestrator.conversation_available()


@pytest.mark.asyncio
async def test_conversational_turn_multi_turn_history_and_critique_propagation(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore
):
    """Verifies that multi-turn dialogue history and prior critique results
    are faithfully passed into the orchestrator/planner on subsequent turns."""
    captured_kwargs = []
    original_run = started_orchestrator.backends[Tier.PLANNER].run

    async def tracking_run(**kwargs):
        captured_kwargs.append(dict(kwargs))
        return await original_run(**kwargs)

    started_orchestrator.backends[Tier.PLANNER].run = tracking_run

    state = store.create()

    # Turn 1: No prior history, no prior critique
    await flows.conversational_turn(started_orchestrator, store, state.session_id, "Create a dark crypto wallet")
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["conversation_history"] == []
    assert captured_kwargs[0]["prior_critique"] is None

    # Finalize & Critique pass
    await flows.finalize(started_orchestrator, store, state.session_id)
    await flows.critique_pass(started_orchestrator, store, state.session_id)

    # Turn 2: Prior history (user + planner) and prior critique should be passed
    await flows.conversational_turn(started_orchestrator, store, state.session_id, "Make the buttons neon green")
    assert len(captured_kwargs) == 2
    turn2_kwargs = captured_kwargs[1]
    assert len(turn2_kwargs["conversation_history"]) == 2  # user + planner from turn 1
    assert turn2_kwargs["conversation_history"][0]["role"] == "user"
    assert turn2_kwargs["conversation_history"][0]["content"] == "Create a dark crypto wallet"
    assert turn2_kwargs["prior_critique"] is not None
    assert "overall_score" in turn2_kwargs["prior_critique"]


@pytest.mark.asyncio
async def test_conversational_turn_updates_constraints_from_planner_delta(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore
):
    """Verifies that structured constraint updates in the planner's JSON delta
    are merged directly into state.constraints."""
    async def delta_planner_run(**kwargs):
        return {
            "tier": "planner",
            "reply_text": "Updated design to dark glassmorphism style.",
            "design_state_delta": {
                "stage": "sketching",
                "constraint_updates": {
                    "style": "glassmorphism dark",
                    "palette": ["#0f172a", "#38bdf8", "#22c55e"],
                    "layout_hints": "centered hero card with tab bar",
                    "locked_regions": [{"bbox": [0.1, 0.1, 0.8, 0.2], "reason": "header brand identity"}],
                },
                "tool_call": None,
                "reasoning_note": "Applying glassmorphism styling.",
            },
        }

    started_orchestrator.backends[Tier.PLANNER].run = delta_planner_run

    state = store.create()
    updated_state = await flows.conversational_turn(
        started_orchestrator, store, state.session_id, "Design a glassy dark card"
    )

    assert updated_state.constraints.style == "glassmorphism dark"
    assert updated_state.constraints.palette == ["#0f172a", "#38bdf8", "#22c55e"]
    assert updated_state.constraints.layout_hints == "centered hero card with tab bar"
    assert len(updated_state.constraints.locked_regions) == 1
    assert updated_state.constraints.locked_regions[0].reason == "header brand identity"
    assert updated_state.constraints.locked_regions[0].bbox == (0.1, 0.1, 0.8, 0.2)


@pytest.mark.asyncio
async def test_finalize_synthesizes_prompt_from_history_and_constraints(
    started_orchestrator: SwapOrchestrator, store: DesignStateStore, sketch_ready_state
):
    """Verifies that flows.finalize synthesizes an enriched prompt from user turns
    and active constraints when prompt=None."""
    captured_prompts = []
    original_req_finalize = started_orchestrator.request_finalize

    async def tracking_finalize(**kwargs):
        captured_prompts.append(kwargs.get("prompt"))
        return await original_req_finalize(**kwargs)

    started_orchestrator.request_finalize = tracking_finalize

    # Set up user turn and constraints
    expected_rev = sketch_ready_state.revision
    sketch_ready_state.append_turn("user", "Fintech mobile dashboard with balance")
    sketch_ready_state.append_turn("planner", "Designing fintech mobile dashboard")
    sketch_ready_state.constraints.style = "minimalist dark"
    sketch_ready_state.constraints.palette = ["#000000", "#00ffcc"]
    sketch_ready_state.constraints.layout_hints = "vertical card stack"
    store.save(sketch_ready_state, expected_rev)

    await flows.finalize(started_orchestrator, store, sketch_ready_state.session_id, prompt=None)

    assert len(captured_prompts) == 1
    prompt_used = captured_prompts[0]
    assert prompt_used is not None
    assert "Fintech mobile dashboard with balance" in prompt_used
    assert "style: minimalist dark" in prompt_used
    assert "palette: #000000, #00ffcc" in prompt_used
    assert "layout: vertical card stack" in prompt_used

