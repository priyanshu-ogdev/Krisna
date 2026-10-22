"""Tests for Phase 2 implementation:
- Tier.POLISH_FAST Consistency Flow Matching (CFM) distillation
- Fast preview flow and HTTP endpoint
- Dual-layer inpainting mask conditioning + Lock Compositor pixel invariance
- Phase 2 ablation configurations and verification runner
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from krivex_inference.orchestrator.model_registry import (
    Tier,
    REGISTRY,
    LOW_VRAM_REGISTRY,
    HIGH_VRAM_REGISTRY,
    get_registry,
    ModelSpec,
)
from krivex_inference.backends.factory import real_backend_factory
from krivex_inference.backends.polish_backend_qwen import QwenImage21Backend
from krivex_inference.orchestrator.service import app
from krivex_training.polish.ablation_runner import (
    load_ablation_config,
    evaluate_ablation_candidate,
    run_ablation,
    run_all_ablations,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_polish_fast_tier_registry():
    """Verify Tier.POLISH_FAST is correctly registered in all VRAM tiers."""
    assert Tier.POLISH_FAST in REGISTRY
    assert Tier.POLISH_FAST in LOW_VRAM_REGISTRY
    assert Tier.POLISH_FAST in HIGH_VRAM_REGISTRY

    fast_spec = REGISTRY[Tier.POLISH_FAST]
    assert fast_spec.vram_gb == 10.0
    assert "Consistency Flow Matching" in fast_spec.quantization

    low_spec = LOW_VRAM_REGISTRY[Tier.POLISH_FAST]
    assert low_spec.vram_gb == 4.5
    assert low_spec.ram_gb == 6.0

    high_spec = HIGH_VRAM_REGISTRY[Tier.POLISH_FAST]
    assert high_spec.vram_gb == 10.0
    assert high_spec.always_resident is True


def test_real_backend_factory_polish_fast(monkeypatch):
    """Verify backend factory configures 6 steps for fast preview tier."""
    monkeypatch.setenv("KRISNA_POLISH_MODEL_ID", "Qwen/Qwen-Image-2.1")
    backend = real_backend_factory(REGISTRY[Tier.POLISH_FAST])
    assert isinstance(backend, QwenImage21Backend)
    assert backend.spec.tier == Tier.POLISH_FAST
    assert backend.num_inference_steps == 6


def test_service_fast_preview_endpoint(client):
    """Test POST /session/{id}/preview endpoint triggering fast preview flow."""
    # 1. Create a session
    res = client.post("/session", json={"style": "modern", "palette": ["#000000", "#FFFFFF"]})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # 2. Add layout so it becomes finalize-eligible
    msg_res = client.post(f"/session/{session_id}/message", json={"message": "create login form"})
    assert msg_res.status_code == 200

    # 3. Trigger preview
    prev_res = client.post(f"/session/{session_id}/preview", json={"prompt": "rapid preview"})
    assert prev_res.status_code == 200
    data = prev_res.json()
    assert data["session_id"] == session_id
    assert data["stage"] == "finalized"
    assert "finalize_output" in data
    assert data["finalize_output"]["image_ref"] is not None


def test_inpaint_mask_generation_and_dual_layer_lock():
    """Verify soft inpaint mask construction and hard Lock Compositor bitwise invariance."""
    backend = QwenImage21Backend(REGISTRY[Tier.POLISH_FAST], strict=False)
    size = (400, 300)
    locked_regions = [
        {"bbox": [0.1, 0.1, 0.3, 0.2]},  # normalized
        {"bbox": [200, 150, 100, 80]},    # absolute
    ]

    mask = backend._build_inpaint_mask(size, locked_regions)
    assert isinstance(mask, Image.Image)
    assert mask.size == size
    assert mask.mode == "L"

    mask_arr = np.array(mask)
    # Outside locked regions should be 255 (editable)
    assert mask_arr[0, 0] == 255
    # Inside normalized region [40..160, 30..90] should be 0 (locked)
    assert mask_arr[50, 80] == 0
    # Inside absolute region [200..300, 150..230] should be 0 (locked)
    assert mask_arr[180, 250] == 0

    # Verify hard lock compositor guarantees 0.00% pixel drift (MSE == 0)
    src_img = Image.new("RGB", size, color=(100, 150, 200))
    # Simulated render with perturbed pixels everywhere
    rendered_img = Image.new("RGB", size, color=(255, 50, 50))

    final_img = backend._apply_locked_regions(src_img, rendered_img, locked_regions)
    final_arr = np.array(final_img)
    src_arr = np.array(src_img)

    # In locked region, final_arr must match src_arr bit-for-bit
    locked_crop_src = src_arr[50:70, 80:120]
    locked_crop_final = final_arr[50:70, 80:120]
    mse = np.mean((locked_crop_src.astype(float) - locked_crop_final.astype(float)) ** 2)
    assert mse == 0.0, f"Expected 0.00% drift on locked region, got MSE={mse}"


def test_ablation_configs_and_runner(tmp_path):
    """Verify all Phase 2 ablation configurations load, validate, and produce receipts."""
    ablation_files = [
        "training/configs/ablations/ablation_a_lora_attn.yaml",
        "training/configs/ablations/ablation_b_dora_attn.yaml",
        "training/configs/ablations/ablation_c_dora_no_wireframe.yaml",
        "training/configs/ablations/ablation_d_dora_no_rag.yaml",
    ]

    for fpath in ablation_files:
        cfg = load_ablation_config(fpath)
        assert "ablation_id" in cfg
        assert "ablation_hypothesis" in cfg
        assert cfg["pretrained_model_name_or_path"] == "Qwen/Qwen-Image-2.1"

        # Run ablation and verify receipt
        receipt = run_ablation(fpath, output_dir=tmp_path)
        assert receipt.status == "COMPLETED"
        assert receipt.metrics.layout_iou > 0.0
        assert receipt.metrics.ocr_readability > 0.0
        assert receipt.metrics.peak_vram_gb <= 12.0

    # Verify CFM distillation config
    cfm_cfg = load_ablation_config("training/configs/cfm_distill_qwen_image21.yaml")
    assert cfm_cfg["distillation_type"] == "consistency_flow_matching"
    assert cfm_cfg["student_target_steps"] == [4, 6, 8]


@pytest.mark.asyncio
async def test_preview_output_tracking():
    from krivex_inference.orchestrator.design_state import DesignState, PreviewOutput
    from krivex_inference.orchestrator.flows import fast_preview
    from krivex_inference.orchestrator.store import DesignStateStore
    from krivex_inference.orchestrator.swap_orchestrator import SwapOrchestrator
    from krivex_inference.orchestrator.vram_budget import VRAMLedger

    state = DesignState()
    assert state.preview_output is None
    # Verify strict wire schema excludes preview_output
    wire = state.to_wire()
    assert "preview_output" not in wire

    state.preview_output = PreviewOutput(
        renderer_used="qwen_image_2_1",
        image_ref="blob://preview123",
        step_count=6,
        prompt_used="modern wallet ui",
    )
    assert state.preview_output.step_count == 6
    assert state.preview_output.image_ref == "blob://preview123"
    # Ensure to_wire still excludes internal/extension fields
    wire_after = state.to_wire()
    assert "preview_output" not in wire_after

    # Run fast preview flow and verify state.preview_output is populated
    store = DesignStateStore()
    orchestrator = SwapOrchestrator(envelope_gb=48.0)
    await orchestrator.start()

    session = store.create(state)
    session.layout.layout_ref = "layout://test/1.json"
    session.layout.wireframe_ref = "blob://wireframe123"
    store.save(session, session.revision)

    finalized = await fast_preview(
        orchestrator=orchestrator,
        store=store,
        session_id=session.session_id,
        prompt="clean fintech screen",
    )

    assert finalized.preview_output is not None
    assert finalized.preview_output.step_count == 6
    assert finalized.preview_output.renderer_used == "qwen_image_2_1"
    assert finalized.preview_output.prompt_used == "clean fintech screen"
    await orchestrator.shutdown()


def test_locked_regions_corner_coordinates_support():
    from PIL import Image
    import numpy as np
    from krivex_inference.backends.polish_backend_qwen import QwenImage21Backend
    from krivex_inference.orchestrator.model_registry import ModelSpec, Tier

    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="Qwen-Image-2.1",
        vram_gb=12.0,
        quantization="bf16",
    )
    backend = QwenImage21Backend(spec, strict=False)

    size = (200, 200)
    # Locked region specified via corner coordinates {"x1", "y1", "x2", "y2"}
    locked_regions = [
        {"x1": 20, "y1": 30, "x2": 100, "y2": 120},
    ]

    mask = backend._build_inpaint_mask(size, locked_regions)
    mask_arr = np.array(mask)
    # Outside should be 255
    assert mask_arr[0, 0] == 255
    # Inside [30..120, 20..100] should be 0 (locked)
    assert mask_arr[50, 50] == 0

    # Composite exact check
    src = Image.new("RGB", size, (40, 50, 60))
    dst = Image.new("RGB", size, (255, 255, 255))
    blended = backend._apply_locked_regions(src, dst, locked_regions)
    blended_arr = np.array(blended)
    src_arr = np.array(src)

    locked_crop_src = src_arr[30:120, 20:100]
    locked_crop_blended = blended_arr[30:120, 20:100]
    mse = np.mean((locked_crop_src.astype(float) - locked_crop_blended.astype(float)) ** 2)
    assert mse == 0.0


def test_generate_ablation_summary_table(tmp_path):
    from krivex_training.polish.ablation_runner import (
        generate_ablation_summary,
        run_all_ablations,
    )

    receipts = run_all_ablations(
        configs_dir="training/configs/ablations",
        output_dir=tmp_path,
    )
    assert len(receipts) >= 4

    summary_file = tmp_path / "ablation_summary.md"
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")

    assert "# Phase 2 Polish Tier Architecture & Conditioning Ablation Matrix" in content
    assert "| Candidate ID | Method / Architecture |" in content
    assert r"\begin{table*}" in content
    assert r"\end{table*}" in content
    assert "A_LORA_ATTN_ONLY" in content
    assert "B_DORA_ATTN_ONLY" in content
    assert "C_NO_WIREFRAME_CONDITION" in content
    assert "D_NO_VISUAL_RAG_GROUNDING" in content

