"""Tests for SketchBackend.run()'s prompt-conditioning wiring.

RESTORED after a regression — this file (and the real-conditioning
implementation it tests) went missing from a working copy partway
through this review. See docs/review/17_sketch_inference_conditioning_and_cfg.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from krisna_inference.backends.sketch_backend import SketchBackend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


def _make_backend_with_fakes(text_embedder, prompt_dim=8):
    spec = ModelSpec(tier=Tier.SKETCH, name="test-sketch", vram_gb=1.0, quantization="none")
    backend = SketchBackend(spec, checkpoint_path="/fake/path", guidance_scale=3.0)

    fake_param = torch.nn.Parameter(torch.zeros(1))
    fake_module = MagicMock()
    fake_module.cfg = MagicMock(prompt_dim=prompt_dim)
    # BUG FOUND ON REVIEW (docs/review/32_model_review_2_sketch.md): a
    # plain `iter([fake_param])` is a ONE-SHOT iterator — every test in
    # this file that calls backend.run() exactly once never noticed, but
    # a test calling run() twice on the same backend instance (needed to
    # test planner_output handling across multiple calls) exhausted it on
    # the second call. next() on an exhausted iterator raises
    # StopIteration — which, raised inside a thread run via
    # asyncio.to_thread, hits a known asyncio/PEP 479 interaction
    # (StopIteration cannot be raised into a Future) that manifests as a
    # HANG, not a clean test failure — very easy to misdiagnose as the
    # code under test hanging rather than the test fixture. Fixed with
    # side_effect so every call gets a fresh iterator.
    fake_module.parameters.side_effect = lambda: iter([fake_param])

    fake_model = MagicMock()
    fake_model.module = fake_module
    fake_model.sample.return_value = {"tokens": [0, 1, 2, 3], "confidence_map": [1.0, 1.0, 1.0, 1.0]}

    backend._model = fake_model
    backend._text_embedder = text_embedder
    backend._loaded = True
    return backend, fake_model


@pytest.fixture(autouse=True)
def _fake_blob_store(monkeypatch):
    import krisna_inference.backends.blob_store_singleton as bss

    store = MagicMock()
    store.load_tokens.return_value = None
    store.save_tokens.side_effect = lambda tokens, prefix=None: f"blob://{prefix}/fake"
    monkeypatch.setattr(bss, "get_blob_store", lambda: store)
    return store


class TestPromptReachesTheModel:
    @pytest.mark.asyncio
    async def test_real_message_is_embedded_and_passed_to_sample(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, fake_model = _make_backend_with_fakes(embedder)

        await backend.run(planner_output=None, message="a settings screen with a dark toggle")

        embedder.embed_text.assert_called_once_with("a settings screen with a dark toggle")
        call_kwargs = fake_model.sample.call_args
        passed_embedding = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs["prompt_embedding"]
        assert torch.equal(passed_embedding, torch.ones(1, 8))
        assert call_kwargs.kwargs.get("guidance_scale", 3.0) == 3.0

    @pytest.mark.asyncio
    async def test_empty_message_falls_back_to_ui_design_text_not_zeros(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        await backend.run(planner_output=None, message="")

        embedder.embed_text.assert_called_once_with("UI design")

    @pytest.mark.asyncio
    async def test_missing_message_kwarg_also_falls_back_to_ui_design(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        await backend.run(planner_output=None)

        embedder.embed_text.assert_called_once_with("UI design")


class TestGracefulDegradationWithoutEmbedder:
    @pytest.mark.asyncio
    async def test_no_embedder_falls_back_to_zero_embedding_and_zero_guidance(self):
        backend, fake_model = _make_backend_with_fakes(text_embedder=None)

        await backend.run(planner_output=None, message="a login screen")

        call_kwargs = fake_model.sample.call_args
        passed_embedding = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs["prompt_embedding"]
        assert torch.equal(passed_embedding, torch.zeros(1, 8))
        assert call_kwargs.kwargs.get("guidance_scale") == 0.0


class TestPromptDimMismatchSafety:
    @pytest.mark.asyncio
    async def test_embedder_dim_mismatch_falls_back_to_zero_and_disables_guidance(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 999)
        backend, fake_model = _make_backend_with_fakes(embedder, prompt_dim=8)

        await backend.run(planner_output=None, message="anything")

        call_kwargs = fake_model.sample.call_args
        passed_embedding = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs["prompt_embedding"]
        assert passed_embedding.shape[-1] == 8
        assert torch.equal(passed_embedding, torch.zeros(1, 8))
        assert call_kwargs.kwargs.get("guidance_scale") == 0.0


class TestPlannerOutputReasoningNoteWiring:
    """Regression tests for a bug found on review
    (docs/review/32_model_review_2_sketch.md): `planner_output` was
    accepted as a parameter but never actually read anywhere in run(),
    the same "accepted but silently ignored" pattern already found once
    in polish_default_backend.py (handoff_image_ref)."""

    @pytest.mark.asyncio
    async def test_reasoning_note_is_folded_into_the_embedded_prompt(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        planner_output = {
            "design_state_delta": {"reasoning_note": "User wants a warmer, friendlier feel."}
        }
        await backend.run(planner_output=planner_output, message="make it cozier")

        embedder.embed_text.assert_called_once()
        prompt_text = embedder.embed_text.call_args[0][0]
        assert "make it cozier" in prompt_text
        assert "User wants a warmer, friendlier feel." in prompt_text

    @pytest.mark.asyncio
    async def test_reasoning_note_is_capped_to_avoid_clip_truncation_risk(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        long_note = "x" * 500
        planner_output = {"design_state_delta": {"reasoning_note": long_note}}
        await backend.run(planner_output=planner_output, message="hi")

        prompt_text = embedder.embed_text.call_args[0][0]
        # The full 500-char note must never appear verbatim — only a
        # bounded slice, so it can't push the user's own message past
        # CLIP's silent 77-token truncation point.
        assert long_note not in prompt_text
        assert "x" * 160 in prompt_text

    @pytest.mark.asyncio
    async def test_none_planner_output_does_not_crash(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        await backend.run(planner_output=None, message="a login screen")
        embedder.embed_text.assert_called_once_with("a login screen")

    @pytest.mark.asyncio
    async def test_dict_shaped_but_unexpected_planner_output_does_not_crash(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        await backend.run(planner_output={"unexpected_shape": True}, message="a login screen")
        embedder.embed_text.assert_called_once_with("a login screen")

    @pytest.mark.asyncio
    async def test_non_dict_planner_output_does_not_crash(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        await backend.run(planner_output="not a dict", message="a login screen")
        embedder.embed_text.assert_called_once_with("a login screen")

    @pytest.mark.asyncio
    async def test_empty_reasoning_note_does_not_add_stray_content(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = torch.ones(1, 8)
        backend, _ = _make_backend_with_fakes(embedder)

        planner_output = {"design_state_delta": {"reasoning_note": "   "}}
        await backend.run(planner_output=planner_output, message="a login screen")

        embedder.embed_text.assert_called_once_with("a login screen")
