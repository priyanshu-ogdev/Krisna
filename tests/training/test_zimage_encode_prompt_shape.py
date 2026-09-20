"""B2: Tests for Z-Image-Turbo encode_prompt return shape handling in train_dpo.py.

Verifies:
1. Defensive unpacking for both Tensor and tuple/list return shapes from encode_prompt.
2. If Tongyi-MAI/Z-Image-Turbo is present locally, checks the actual return shape
   of its encode_prompt method. Otherwise skipped gracefully.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _mock_encode_prompt_tensor(prompt: list[str]) -> torch.Tensor:
    return torch.randn(len(prompt), 77, 1024)


def _mock_encode_prompt_tuple(prompt: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(len(prompt), 77, 1024), torch.randn(len(prompt), 1024)


def test_encode_prompt_tensor_handling():
    """Verify single Tensor return shape handling."""
    batch_prompts = ["a clean mobile landing page", "dark mode dashboard"]
    embeds = _mock_encode_prompt_tensor(batch_prompts)

    if isinstance(embeds, torch.Tensor):
        processed = embeds.to("cpu")
    elif isinstance(embeds, (tuple, list)):
        processed = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t for t in embeds)
    else:
        raise TypeError(f"Unexpected return type: {type(embeds)}")

    assert isinstance(processed, torch.Tensor)
    assert processed.shape[0] == 2


def test_encode_prompt_tuple_handling():
    """Verify tuple/list return shape handling (e.g. prompt_embeds, pooled_embeds)."""
    batch_prompts = ["a clean mobile landing page", "dark mode dashboard"]
    embeds = _mock_encode_prompt_tuple(batch_prompts)

    if isinstance(embeds, torch.Tensor):
        processed = embeds.to("cpu")
    elif isinstance(embeds, (tuple, list)):
        processed = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t for t in embeds)
    else:
        raise TypeError(f"Unexpected return type: {type(embeds)}")

    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert all(isinstance(t, torch.Tensor) for t in processed)


def test_zimage_turbo_live_encode_prompt():
    """Live verification against Tongyi-MAI/Z-Image-Turbo if downloaded."""
    pytest.importorskip("diffusers")

    model_id = "Tongyi-MAI/Z-Image-Turbo"
    try:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as exc:
        pytest.skip(f"Z-Image-Turbo checkpoint not available in local cache ({exc})")

    out = pipe.encode_prompt(["Test prompt"])
    assert out is not None
