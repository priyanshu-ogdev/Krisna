from __future__ import annotations

import json
import sys
import types

import pytest
from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from PIL import Image


def _spec(tier: Tier = Tier.POLISH_DEFAULT) -> ModelSpec:
    return ModelSpec(tier=tier, name="Qwen-Image-2.1", vram_gb=12.0, quantization="bf16")


class _Generator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _LiveScheduler:
    def __init__(self):
        self.timesteps = [1.0, 0.5, 0.0]


class _NativePipeline:
    def __init__(self, image):
        self.scheduler = _LiveScheduler()
        self.image = image
        self.calls = []

    def __call__(self, prompt, image, num_inference_steps, generator=None, true_cfg_scale=1.0):
        self.calls.append(
            {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
            }
        )
        return types.SimpleNamespace(images=[self.image])


@pytest.mark.asyncio
async def test_real_call_uses_only_native_signature_seed_and_live_receipt(monkeypatch):
    fake_torch = types.SimpleNamespace(Generator=_Generator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    image = Image.new("RGB", (32, 32), color=(20, 30, 40))
    pipeline = _NativePipeline(image)
    backend = QwenImage21Backend(_spec(), strict=True, device="cpu")
    backend._pipeline = pipeline
    backend._native_call_parameters = {"self", "prompt", "image", "num_inference_steps", "generator", "true_cfg_scale"}
    backend._pipeline_class_name = "QwenImage21Pipeline"
    backend._loaded = True
    ref = get_blob_store().save_image(image, prefix="qwen_native_contract")

    result = await backend.run(
        handoff_image_ref=ref,
        prompt="native contract",
        num_inference_steps=3,
        seed=0,
        true_cfg_scale=1.25,
    )

    assert set(pipeline.calls[0]) == {"prompt", "image", "num_inference_steps", "generator", "true_cfg_scale"}
    assert pipeline.calls[0]["generator"].seed == 0
    assert result["synthetic"] is False
    assert result["strength"] is None
    assert result["run_receipt"]["scheduler_class"] == "_LiveScheduler"
    assert result["run_receipt"]["actual_steps"] == 3
    assert result["run_receipt"]["actual_timesteps"] == [1.0, 0.5, 0.0]
    assert result["timesteps"] == [1.0, 0.5, 0.0]


@pytest.mark.asyncio
async def test_failed_non_strict_real_call_is_labeled_synthetic(monkeypatch):
    class FailingPipeline(_NativePipeline):
        def __call__(self, prompt, image, num_inference_steps, generator=None, true_cfg_scale=1.0):
            raise RuntimeError("mock pipeline failure")

    image = Image.new("RGB", (32, 32), color=(20, 30, 40))
    backend = QwenImage21Backend(_spec(), strict=False)
    backend._pipeline = FailingPipeline(image)
    backend._native_call_parameters = {"self", "prompt", "image", "num_inference_steps", "generator", "true_cfg_scale"}
    backend._pipeline_class_name = "QwenImage21Pipeline"
    backend._loaded = True
    ref = get_blob_store().save_image(image, prefix="qwen_synthetic_fallback")

    result = await backend.run(handoff_image_ref=ref)

    assert result["synthetic"] is True
    assert result["runtime_mode"] == "synthetic"
    assert "mock pipeline failure" in result["run_receipt"]["fallback_reason"]
    assert result["run_receipt"]["scheduler_source"] == "synthetic_compatibility"


def test_strict_manifest_requires_exact_revision_before_lora_load(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "base_model": "Qwen/Qwen-Image-2.1",
                "pipeline": "QwenImage21Pipeline",
                "conditioning": "image",
                "base_revision": "main",
                "dataset_manifest_sha256": "a" * 64,
                "split_counts": {"train": 1},
                "lora_target_modules": ["to_q"],
            }
        )
    )
    backend = QwenImage21Backend(_spec(), lora_adapter_path=str(adapter), strict=True)

    class Pipe:
        loaded = False

        def load_lora_weights(self, path):
            self.loaded = True

    pipe = Pipe()
    with pytest.raises(RuntimeError, match="Invalid Qwen LoRA adapter manifest"):
        backend._validate_and_load_lora(pipe, str(adapter))
    assert pipe.loaded is False


def test_quality_is_frozen_even_for_explicit_adapter(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        QwenImage21Backend(_spec(Tier.POLISH_QUALITY), lora_adapter_path=str(tmp_path), strict=False)


def test_locked_regions_are_half_open_and_reject_ambiguous_input():
    backend = QwenImage21Backend(_spec(), strict=False)
    source = Image.new("RGB", (10, 10), color=(255, 0, 0))
    rendered = Image.new("RGB", (10, 10), color=(0, 255, 0))
    result = backend._apply_locked_regions(source, rendered, [{"bbox": [2, 2, 3, 3]}])
    assert result.getpixel((4, 4)) == (255, 0, 0)
    assert result.getpixel((5, 5)) == (0, 255, 0)
    with pytest.raises(ValueError, match="mixes normalized and absolute"):
        backend._apply_locked_regions(source, rendered, [{"bbox": [0.2, 0.2, 3, 3]}])
