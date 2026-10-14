import json
import sys
import types
import pytest
from PIL import Image

from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.backends.polish_backend_qwen import QWEN_IMAGE_21_REVISION
from krisna_inference.backends.blob_store_singleton import get_blob_store


@pytest.mark.asyncio
async def test_polish_quality_steps_and_strength():
    spec = ModelSpec(
        tier=Tier.POLISH_QUALITY,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, edit_strength=0.7, strict=False)
    await backend.load()

    store = get_blob_store()
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    ref = store.save_image(img, prefix="test_input")

    res = await backend.run(
        handoff_image_ref=ref,
        prompt="e-commerce product screen",
        quality=True,
    )

    assert res["tier"] == "polish_quality"
    assert res["steps"] == 40  # Quality tier forces 40 steps
    assert res["strength"] == 0.7
    assert res["renderer"] == "qwen_image_2_1"

    await backend.unload()


@pytest.mark.asyncio
async def test_polish_backend_raises_when_unloaded():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)
    with pytest.raises(RuntimeError, match="not loaded"):
        await backend.run(prompt="test")


@pytest.mark.asyncio
async def test_strict_mode_rejects_missing_condition_instead_of_synthesizing():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=True)
    backend._loaded = True
    with pytest.raises(ValueError, match="requires a valid image conditioning"):
        await backend.run(handoff_image_ref=None)
    await backend.unload()


@pytest.mark.asyncio
async def test_strict_mode_rejects_unavailable_pipeline():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=True)
    backend._loaded = True
    store = get_blob_store()
    ref = store.save_image(Image.new("RGB", (32, 32)), prefix="test_real_mode")
    with pytest.raises(RuntimeError, match="pipeline is unavailable"):
        await backend.run(handoff_image_ref=ref)
    await backend.unload()


@pytest.mark.asyncio
async def test_strict_load_requires_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    fake_diffusers = types.SimpleNamespace(QwenImagePipeline=object)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=True, device="cuda")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        await backend.load()


@pytest.mark.asyncio
async def test_strict_load_does_not_fallback_to_compatible_pipeline(monkeypatch):
    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        float32=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    fake_diffusers = types.SimpleNamespace(QwenImagePipeline=object)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=True, device="cuda")
    with pytest.raises(RuntimeError, match="QwenImage21Pipeline is required"):
        await backend.load()


@pytest.mark.asyncio
async def test_strict_load_rejects_pipeline_error_without_synthetic_fallback(monkeypatch):
    class NativePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ValueError("checkpoint is unavailable")

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        float32=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    fake_diffusers = types.SimpleNamespace(QwenImage21Pipeline=NativePipeline)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=True, device="cuda")
    with pytest.raises(RuntimeError, match="checkpoint is unavailable"):
        await backend.load()
    assert backend._pipeline is None
    assert backend.is_loaded is False


def test_strict_mode_rejects_unpinned_revision():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    with pytest.raises(ValueError, match="pinned model revision"):
        QwenImage21Backend(spec, strict=True, revision="main")
    assert QWEN_IMAGE_21_REVISION == "d26bb61231c349cf6b7896fa83353113880e1ba3"


def test_lora_manifest_verification(tmp_path):
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    adapter_dir = tmp_path / "mock_lora"
    adapter_dir.mkdir()
    manifest_file = adapter_dir / "manifest.json"
    manifest_file.write_text(json.dumps({"run_id": "exp_dreambooth_01", "base_model": "Qwen-Image-2.1"}))

    backend = QwenImage21Backend(spec, lora_adapter_path=str(adapter_dir), strict=False)
    assert backend.lora_adapter_path == str(adapter_dir)


def test_flow_ode_scheduler_timesteps_and_guidance():
    from krisna_inference.backends.polish_backend_qwen import FlowODEScheduler

    sched = FlowODEScheduler(num_steps=10, strength=0.5, guidance_scale=4.0, guidance_schedule="cosine_decay")
    steps = sched.timesteps()
    assert len(steps) == 11
    assert steps[0] == pytest.approx(0.5)
    assert steps[-1] == pytest.approx(1.0)

    # Dynamic guidance decay: early steps have higher guidance than late steps
    w_early = sched.guidance_at(0.0)
    w_late = sched.guidance_at(1.0)
    assert w_early > w_late
    assert w_early == pytest.approx(4.0)
    assert w_late == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_polish_locked_regions_preserve_pixels_exactly():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)
    await backend.load()

    store = get_blob_store()
    img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    ref = store.save_image(img, prefix="test_locked")

    res = await backend.run(
        handoff_image_ref=ref,
        prompt="dashboard card with navbar",
        locked_regions=[{"x": 10, "y": 10, "w": 80, "h": 30}],
    )

    assert res["renderer"] == "qwen_image_2_1"
    assert "timesteps" in res
    assert len(res["timesteps"]) > 0
    await backend.unload()
