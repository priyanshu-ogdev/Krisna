from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.layout_backend import LayoutBackend
from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


# ---------------------------------------------------------------------------
# LayoutBackend (Sketch Tier) Lifecycle and Generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layout_backend_lifecycle_and_generation(tmp_path):
    spec = ModelSpec(
        tier=Tier.SKETCH,
        name="Krisna structured Layout IR",
        vram_gb=0.2,
        quantization="deterministic renderer",
        always_resident=True,
    )
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    assert not backend.is_loaded

    await backend.load()
    assert backend.is_loaded

    planner_output = {
        "design_state_delta": {
            "stage": "sketching",
            "constraint_updates": {
                "style": "clean modern SaaS",
                "palette": ["#ffffff", "#000000", "#4f46e5"],
                "layout_hints": "navbar on top, hero card in center",
            },
            "reasoning_note": "Emphasizing clean layout hierarchy.",
        }
    }

    result = await backend.run(
        planner_output=planner_output,
        message="Create a signup page",
        session_id="sess_123",
    )

    assert result["tier"] == "sketch"
    assert result["mode"] in ("layout_baseline", "neural_layout_flow")
    assert "layout_ir" in result
    assert result["layout_ref"].endswith(".json")
    assert result["wireframe_ref"].startswith("blob://")
    assert Path(result["layout_ref"]).exists()

    await backend.unload()
    assert not backend.is_loaded


@pytest.mark.asyncio
async def test_layout_backend_raises_when_unloaded():
    spec = ModelSpec(
        tier=Tier.SKETCH,
        name="Krisna structured Layout IR",
        vram_gb=0.2,
        quantization="deterministic renderer",
        always_resident=True,
    )
    backend = LayoutBackend(spec)
    with pytest.raises(RuntimeError, match="not loaded"):
        await backend.run()


# ---------------------------------------------------------------------------
# QwenImage21Backend (Polish Tier) Lifecycle and Conditioning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_qwen_image_21_backend_conditioning_synthetic(tmp_path):
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, edit_strength=0.65, strict=False)
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
