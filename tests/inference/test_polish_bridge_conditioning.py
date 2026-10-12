import pytest
from PIL import Image

from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.backends.blob_store_singleton import get_blob_store


@pytest.mark.asyncio
async def test_qwen_image_21_backend_conditioning_synthetic(tmp_path):
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, edit_strength=0.65)
    await backend.load()

    store = get_blob_store()
    sketch_img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    sketch_ref = store.save_image(sketch_img, prefix="test_wireframe")

    res = await backend.run(
        handoff_image_ref=sketch_ref,
        prompt="clean modern SaaS dashboard",
        constraints={
            "style": "clean modern SaaS",
            "locked_regions": [{"bbox": [0.0, 0.0, 1.0, 0.2], "reason": "navbar"}],
        },
        quality=False,
    )

    assert res["tier"] == "polish_default"
    assert res["image_ref"].startswith("blob://")
    assert res["renderer"] == "qwen_image_2_1"
    assert res["steps"] == 28
    assert res["strength"] == 0.65

    # Check verifier scores attached
    scores = res["verifier_scores"]
    assert "clip_alignment" in scores
    assert "layout_iou" in scores

    await backend.unload()
    assert not backend.is_loaded
