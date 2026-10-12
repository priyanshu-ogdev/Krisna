import pytest
from pathlib import Path

from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.backends.layout_backend import LayoutBackend


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
    assert result["mode"] == "layout_baseline"
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
