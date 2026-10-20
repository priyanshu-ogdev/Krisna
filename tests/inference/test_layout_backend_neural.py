"""Tests for LayoutBackend Neuro-Symbolic Production Engine.

Grounding:
- Badros, G. J., et al. (ACM TOCHI 2001). "The Cassowary Linear Arithmetic Constraint Solving Algorithm."
- Huang, C., et al. (NeurIPS 2023). "Composer: Creative and Controllable Image Synthesis with Composable Conditions."
- Park, T., et al. (IEEE/CVF CVPR 2019 Oral). "Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)."
"""

from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image

from krisna_inference.backends.layout_backend import LayoutBackend
from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.planner_schema import AddComponentAction, LockRegionAction
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


@pytest.mark.asyncio
async def test_layout_backend_production_execution(tmp_path):
    """Verify LayoutBackend executes deterministic constraint satisfaction and outputs valid LayoutIR and wireframe."""
    spec = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    await backend.load()

    constraints = {
        "style": "clean dark",
        "palette": ["#0f172a", "#38bdf8"],
        "actions": [
            AddComponentAction(
                component_type="card",
                bbox=[0.10, 0.25, 0.80, 0.50],
                target_id="analytics_card",
                props={"label": "Traffic Overview"},
            ),
            AddComponentAction(
                component_type="button",
                bbox=[0.70, 0.85, 0.20, 0.08],
                target_id="export_btn",
                props={"label": "Download CSV"},
            ),
            LockRegionAction(target_id="title"),
        ],
    }

    result = await backend.run(
        planner_output={},
        constraints=constraints,
        message="A SaaS dashboard with analytics",
        session_id="test_sess",
        turn_id="turn_1",
    )

    assert result["mode"] == "layout_baseline"
    assert "flow_metrics" in result
    metrics = result["flow_metrics"]

    assert metrics["trajectory_steps"] == 1
    assert metrics["element_count"] > 0

    # Verify LayoutIR integrity
    layout_ir = result["layout_ir"]
    node_ids = [n["id"] for n in layout_ir["nodes"]]
    assert "analytics_card" in node_ids
    assert "export_btn" in node_ids
    assert "title" in node_ids

    # Verify locked element immutability
    title_node = next(n for n in layout_ir["nodes"] if n["id"] == "title")
    assert title_node["locked"] is True

    # Verify wireframe was generated and saved
    wireframe_ref = result["wireframe_ref"]
    assert wireframe_ref.startswith("blob://")
    img = get_blob_store().load_image(wireframe_ref)
    assert img.size == (layout_ir["canvas_width"], layout_ir["canvas_height"])


@pytest.mark.asyncio
async def test_layout_backend_baseline_mode(tmp_path):
    """Verify LayoutBackend provides deterministic layout with zero displacement."""
    spec = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    await backend.load()

    result = await backend.run(
        planner_output={},
        constraints={"style": "minimal"},
        message="Minimal layout",
        session_id="baseline_sess",
        turn_id="turn_1",
    )

    assert result["mode"] == "layout_baseline"
    assert result["flow_metrics"]["trajectory_steps"] == 1
    assert result["flow_metrics"]["displacement_mean"] == 0.0
    assert result["flow_metrics"]["element_count"] == len(result["layout_ir"]["nodes"])


@pytest.mark.asyncio
async def test_layout_backend_with_transparent_asset_mode(tmp_path):
    """Verify LayoutBackend preserves native RGBA mode end-to-end for transparent asset requests."""
    spec = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    await backend.load()

    result = await backend.run(
        planner_output={},
        constraints={"asset_mode": "transparent", "style": "modern"},
        message="A transparent component badge",
        session_id="trans_sess",
        turn_id="turn_1",
    )

    wireframe_ref = result["wireframe_ref"]
    img = get_blob_store().load_image(wireframe_ref)
    assert img.mode == "RGBA"


@pytest.mark.asyncio
async def test_layout_backend_generates_and_persists_segmentation_map(tmp_path):
    """Verify LayoutBackend produces a canonical semantic segmentation map in BlobStore (NVIDIA SPADE style)."""
    from krisna_inference.layout_renderer import UI_SEMANTIC_PALETTE

    spec = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    backend = LayoutBackend(spec, output_dir=str(tmp_path))
    await backend.load()

    result = await backend.run(
        planner_output={},
        constraints={
            "style": "dark",
            "actions": [
                AddComponentAction(component_type="button", bbox=[0.7, 0.8, 0.2, 0.1], target_id="btn_submit"),
                AddComponentAction(component_type="card", bbox=[0.1, 0.2, 0.8, 0.5], target_id="main_card"),
                LockRegionAction(target_id="title"),
            ],
        },
        message="A dashboard layout",
        session_id="seg_test_sess",
        turn_id="turn_1",
    )

    assert "segmentation_ref" in result
    seg_ref = result["segmentation_ref"]
    assert seg_ref.startswith("blob://")

    seg_img = get_blob_store().load_image(seg_ref)
    assert seg_img.size == (result["layout_ir"]["canvas_width"], result["layout_ir"]["canvas_height"])
    assert seg_img.mode == "RGB"

    # Check that colors in the segmentation image match UI_SEMANTIC_PALETTE
    palette_counts = seg_img.getcolors(maxcolors=65536)
    assert palette_counts is not None
    colors = {c[1] for c in palette_counts}
    assert UI_SEMANTIC_PALETTE["background"] in colors
    assert UI_SEMANTIC_PALETTE["button"] in colors
    assert UI_SEMANTIC_PALETTE["card"] in colors
    assert UI_SEMANTIC_PALETTE["locked"] in colors


@pytest.mark.asyncio
async def test_polish_backend_consumes_segmentation_map(tmp_path):
    """Verify QwenImage21Backend consumes segmentation_ref and logs NVIDIA SPADE multi-stream telemetry."""
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend

    spec_sketch = ModelSpec(tier=Tier.SKETCH, name="LayoutBaseline", vram_gb=0.0, quantization="none")
    sketch_backend = LayoutBackend(spec_sketch, output_dir=str(tmp_path))
    await sketch_backend.load()

    sketch_result = await sketch_backend.run(
        planner_output={},
        constraints={"style": "clean"},
        message="Test page",
        session_id="polish_seg_sess",
        turn_id="turn_1",
    )

    spec_polish = ModelSpec(tier=Tier.POLISH_DEFAULT, name="Qwen-Image-2.1", vram_gb=4.5, quantization="BF16", always_resident=False)
    polish_backend = QwenImage21Backend(spec_polish)
    await polish_backend.load()

    polish_result = await polish_backend.run(
        handoff_image_ref=sketch_result["wireframe_ref"],
        segmentation_ref=sketch_result["segmentation_ref"],
        prompt="A clean UI with polished cards",
        constraints={"style": "clean"},
        locked_regions=[],
    )

    receipt = polish_result["run_receipt"]
    assert receipt["segmentation_used"] is True
    assert receipt["segmentation_ref"] == sketch_result["segmentation_ref"]
    assert receipt["spatial_conditioning_mode"] == "NVIDIA_SPADE_MultiStream"
