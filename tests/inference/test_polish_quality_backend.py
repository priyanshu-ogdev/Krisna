import json
import pytest
from PIL import Image

from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.backends.blob_store_singleton import get_blob_store


@pytest.mark.asyncio
async def test_polish_quality_steps_and_strength():
    spec = ModelSpec(
        tier=Tier.POLISH_QUALITY,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, edit_strength=0.7)
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
    backend = QwenImage21Backend(spec)
    with pytest.raises(RuntimeError, match="not loaded"):
        await backend.run(prompt="test")


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

    backend = QwenImage21Backend(spec, lora_adapter_path=str(adapter_dir))
    assert backend.lora_adapter_path == str(adapter_dir)
