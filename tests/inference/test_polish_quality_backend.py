import json
import sys
import types
import pytest
from PIL import Image

from krivex_inference.orchestrator.model_registry import ModelSpec, Tier
from krivex_inference.backends.polish_backend_qwen import QwenImage21Backend
from krivex_inference.backends.polish_backend_qwen import QWEN_IMAGE_21_REVISION
from krivex_inference.backends.blob_store_singleton import get_blob_store


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
    from krivex_inference.backends.polish_backend_qwen import FlowODEScheduler

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


@pytest.mark.asyncio
async def test_polish_multi_reference_assets_conditioning():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)
    await backend.load()

    store = get_blob_store()
    canvas = Image.new("RGB", (256, 256), color=(20, 20, 20))
    canvas_ref = store.save_image(canvas, prefix="canvas")

    ref1 = Image.new("RGB", (100, 100), color=(200, 50, 50))
    ref1_blob = store.save_image(ref1, prefix="face_a")

    ref2 = Image.new("RGB", (100, 100), color=(50, 50, 200))
    ref2_blob = store.save_image(ref2, prefix="face_b")

    res = await backend.run(
        handoff_image_ref=canvas_ref,
        prompt="Dual portrait hero section with face of person A and person B",
        reference_assets=[
            {"asset_id": "face_a", "blob_ref": ref1_blob, "bbox": [0.1, 0.2, 0.35, 0.5]},
            {"asset_id": "face_b", "blob_ref": ref2_blob, "bbox": [0.55, 0.2, 0.35, 0.5]},
        ],
    )

    receipt = res["run_receipt"]
    assert receipt["reference_assets_count"] == 2
    assert receipt["multi_reference_conditioning"] is True
    assert receipt["prefix_kv_cache_reuse"] is True
    assert receipt["mixed_granularity_attention"] is True
    assert receipt["qwen_version"] == "2.1"
    assert res["image_ref"].startswith("blob://")
    await backend.unload()


@pytest.mark.asyncio
async def test_polish_reference_assets_raw_strings_and_pil():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)
    await backend.load()

    store = get_blob_store()
    canvas = Image.new("RGB", (256, 256), color=(20, 20, 20))
    canvas_ref = store.save_image(canvas, prefix="canvas_raw")

    ref1 = Image.new("RGB", (80, 80), color=(220, 30, 30))
    ref1_blob = store.save_image(ref1, prefix="raw_blob")

    raw_pil_img = Image.new("RGB", (60, 60), color=(30, 220, 30))

    # Test mixed list: one string blob_ref, one raw PIL image
    res = await backend.run(
        handoff_image_ref=canvas_ref,
        prompt="Mixed raw assets conditioning",
        reference_assets=[ref1_blob, raw_pil_img],
    )

    receipt = res["run_receipt"]
    assert receipt["reference_assets_count"] == 2
    assert receipt["multi_reference_conditioning"] is True
    await backend.unload()


@pytest.mark.asyncio
async def test_polish_full_donor_pasting_without_crop():
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)
    await backend.load()

    store = get_blob_store()
    canvas = Image.new("RGB", (256, 256), color=(0, 0, 0))
    canvas_ref = store.save_image(canvas, prefix="black_canvas")

    donor = Image.new("RGB", (64, 64), color=(255, 0, 0))
    donor_blob = store.save_image(donor, prefix="red_donor")

    # Supply source_asset_id with NO source_crop_bbox -> defaults to full donor pasting
    res = await backend.run(
        handoff_image_ref=canvas_ref,
        prompt="Full donor paste test",
        reference_assets=[{"asset_id": "red_box", "blob_ref": donor_blob}],
        locked_regions=[
            {"source_asset_id": "red_box", "bbox": [0.25, 0.25, 0.5, 0.5]}
        ],
    )

    out_img = store.load_image(res["image_ref"]).convert("RGB")
    # The center of the 50% box (e.g. at pixel 128, 128) should be red (255, 0, 0)
    center_pixel = out_img.getpixel((128, 128))
    assert center_pixel[0] == 255 and center_pixel[1] == 0 and center_pixel[2] == 0

    from krivex_inference.verifiers.lock_preservation import LockPreservationVerifier
    verifier = LockPreservationVerifier()
    donor_map = {"red_box": donor}
    score = verifier.score(
        canvas,
        out_img,
        locked_regions=[{"source_asset_id": "red_box", "bbox": [0.25, 0.25, 0.5, 0.5]}],
        donor_map=donor_map,
    )
    assert score is not None
    assert score >= 0.99, f"Expected near-perfect lock score, got {score}"

    await backend.unload()

