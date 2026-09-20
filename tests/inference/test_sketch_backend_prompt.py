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
    fake_module.parameters.return_value = iter([fake_param])

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
