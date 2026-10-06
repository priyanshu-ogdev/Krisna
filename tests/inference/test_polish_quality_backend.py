"""Tests for QwenImageEditBackend — previously had NO direct test
coverage at all (docs/review/36_model_review_4_polish_quality.md),
despite its B3 runtime detection logic being exactly the kind of subtle,
easy-to-silently-break behavior that deserves one. Added after
definitively confirming (against the real, currently-pinned diffusers
commit) that `strength` does not exist in QwenImageEditPlusPipeline's
real `__call__` signature — these tests lock in that the backend
degrades correctly either way, not just when the detection happens to
fire.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("diffusers")

from krisna_inference.backends.polish_quality_backend import (
    QwenImageEditBackend,
    _edit_instruction_from_constraints,
)
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


def _make_fake_pipe(has_strength: bool):
    """A MagicMock whose __call__ has a REAL (not mocked) signature, so
    inspect.signature() — the actual mechanism B3 uses — behaves exactly
    like it would against a real pipeline."""
    if has_strength:
        def fake_call(self, image=None, prompt=None, strength=0.6, true_cfg_scale=4.0,
                       num_inference_steps=40):
            pass
    else:
        def fake_call(self, image=None, prompt=None, true_cfg_scale=4.0,
                       num_inference_steps=40):
            pass

    pipe = MagicMock()
    pipe.__call__ = fake_call.__get__(pipe)
    pipe.to.return_value = pipe
    return pipe


class TestB3StrengthDetection:
    @pytest.mark.asyncio
    async def test_edit_strength_set_to_none_when_pipeline_lacks_strength_param(self):
        """The CONFIRMED real-world case: QwenImageEditPlusPipeline, as
        actually pinned in requirements-inference.txt, has no `strength`
        parameter — this must degrade to None, not crash."""
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY])
        fake_pipe = _make_fake_pipe(has_strength=False)

        with patch("diffusers.QwenImageEditPlusPipeline.from_pretrained", return_value=fake_pipe), \
             patch("diffusers.BitsAndBytesConfig"):
            await backend.load()

        assert backend.edit_strength is None

    @pytest.mark.asyncio
    async def test_edit_strength_preserved_when_pipeline_has_strength_param(self):
        """Forward-compatibility case: a future diffusers version that
        DOES add `strength` must have it picked up automatically, with
        no code change needed — this is the whole point of doing this
        as a runtime check rather than a hardcoded skip."""
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY], edit_strength=0.65)
        fake_pipe = _make_fake_pipe(has_strength=True)

        with patch("diffusers.QwenImageEditPlusPipeline.from_pretrained", return_value=fake_pipe), \
             patch("diffusers.BitsAndBytesConfig"):
            await backend.load()

        assert backend.edit_strength == 0.65


class TestRunRequiresHandoffImage:
    @pytest.mark.asyncio
    async def test_raises_when_handoff_image_ref_missing(self):
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY])
        backend._loaded = True
        with pytest.raises(ValueError, match="requires handoff_image_ref"):
            await backend.run(handoff_image_ref=None)

    @pytest.mark.asyncio
    async def test_raises_when_not_loaded(self):
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY])
        with pytest.raises(RuntimeError, match="not loaded"):
            await backend.run(handoff_image_ref="blob://fake")


class TestRunPassesStrengthOnlyWhenAvailable:
    @pytest.mark.asyncio
    async def test_strength_omitted_from_pipe_call_when_none(self):
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY])
        backend._loaded = True
        backend.edit_strength = None  # as B3 would set it for the real pinned pipeline

        fake_image = MagicMock()
        fake_pipe_result = MagicMock()
        fake_pipe_result.images = [fake_image]
        backend._pipe = MagicMock(return_value=fake_pipe_result)

        fake_store = MagicMock()
        fake_store.load_image.return_value = fake_image
        fake_store.save_image.return_value = "blob://out/fake"

        with patch(
            "krisna_inference.backends.polish_quality_backend.get_blob_store",
            return_value=fake_store,
        ):
            await backend.run(handoff_image_ref="blob://in/fake", prompt="refine this")

        call_kwargs = backend._pipe.call_args.kwargs
        assert "strength" not in call_kwargs

    @pytest.mark.asyncio
    async def test_strength_included_when_set(self):
        backend = QwenImageEditBackend(REGISTRY[Tier.POLISH_QUALITY], edit_strength=0.7)
        backend._loaded = True
        backend.edit_strength = 0.7  # as B3 would leave it for a future, strength-supporting pipeline

        fake_image = MagicMock()
        fake_pipe_result = MagicMock()
        fake_pipe_result.images = [fake_image]
        backend._pipe = MagicMock(return_value=fake_pipe_result)

        fake_store = MagicMock()
        fake_store.load_image.return_value = fake_image
        fake_store.save_image.return_value = "blob://out/fake"

        with patch(
            "krisna_inference.backends.polish_quality_backend.get_blob_store",
            return_value=fake_store,
        ):
            await backend.run(handoff_image_ref="blob://in/fake", prompt="refine this")

        call_kwargs = backend._pipe.call_args.kwargs
        assert call_kwargs["strength"] == 0.7


class TestEditInstructionFromConstraints:
    def test_default_style_used_when_absent(self):
        instruction = _edit_instruction_from_constraints({}, None)
        assert "clean modern UI design" in instruction

    def test_custom_style_used_when_present(self):
        instruction = _edit_instruction_from_constraints({"style": "brutalist"}, None)
        assert "brutalist" in instruction

    def test_layout_hints_appended(self):
        instruction = _edit_instruction_from_constraints(
            {"layout_hints": "sidebar on the left"}, None
        )
        assert "sidebar on the left" in instruction

    def test_locked_regions_appended_as_instruction_text(self):
        """This is the ONLY region-locking mechanism currently in
        effect — confirmed no mask-based alternative exists in the real
        pinned pipeline, see this backend's own updated docstring."""
        locked = [{"reason": "logo must stay fixed"}, {"reason": "nav bar unchanged"}]
        instruction = _edit_instruction_from_constraints({}, locked)
        assert "logo must stay fixed" in instruction
        assert "nav bar unchanged" in instruction

    def test_no_locked_regions_text_when_absent(self):
        instruction = _edit_instruction_from_constraints({}, None)
        assert "locked regions" not in instruction
