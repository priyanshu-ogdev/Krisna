"""Tests for Phase 4: Consistency Flow Matching (CFM) Distillation, Fast Preview, and Step Pareto Gates.
Theoretical Grounding:
- Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", Springer ECCV 2024.
- Song et al., "Consistency Models", ICML 2023.
- Song & Dhariwal, "Improved Techniques for Training Consistency Models", ICLR 2024.
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from krisna_inference.orchestrator import flows
from krisna_inference.orchestrator.design_state import SessionStage
from krisna_inference.orchestrator.model_registry import (
    HIGH_VRAM_REGISTRY,
    LOW_VRAM_REGISTRY,
    REGISTRY,
    MockBackend,
    Tier,
)
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator
from scripts.training.compare_cfm_steps import (
    evaluate_step_pareto,
    generate_cfm_distillation_receipt,
)


class TestPhase4CFMConfig:
    def test_cfm_distill_yaml_validity(self):
        cfg_path = Path("training/configs/cfm_distill_qwen_image21.yaml")
        assert cfg_path.exists(), "cfm_distill_qwen_image21.yaml must exist"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        assert data["teacher_model_name_or_path"] == "Qwen/Qwen-Image-2.1"
        assert data["distillation_type"] == "consistency_flow_matching"
        assert data["student_target_steps"] == [4, 6, 8]
        assert data["loss_type"] == "huber"
        assert data["huber_c"] == 0.001
        assert data["ema_decay"] == 0.999
        assert data["rank"] == 32
        assert data["lora_alpha"] == 16
        assert data["use_dora"] is True
        assert data["mixed_precision"] == "bf16"

        # Gate targets
        targets = data["target_metrics"]
        assert targets["max_latency_seconds_a6000"] <= 1.0
        assert targets["max_latency_seconds_consumer"] <= 3.0
        assert targets["min_layout_iou"] >= 0.85
        assert targets["min_ocr_readability"] >= 0.90

    def test_launcher_script_exists(self):
        launcher_path = Path("scripts/training/train_polish_fast_preview.sh")
        assert launcher_path.exists(), "train_polish_fast_preview.sh launcher must exist"
        content = launcher_path.read_text(encoding="utf-8")
        assert "cfm_distill_qwen_image21.yaml" in content
        assert "compare_cfm_steps.py" in content


class TestPhase4ParetoEvaluation:
    def test_step_pareto_metrics_monotonicity(self):
        results = evaluate_step_pareto([4, 6, 8, 16, 30])
        assert len(results) == 5

        # Check latency monotonicity
        steps = [4, 6, 8, 16, 30]
        latencies_a6000 = [results[f"steps_{s}"]["latency_seconds_a6000"] for s in steps]
        latencies_consumer = [results[f"steps_{s}"]["latency_seconds_consumer"] for s in steps]
        ious = [results[f"steps_{s}"]["layout_iou"] for s in steps]

        assert latencies_a6000 == sorted(latencies_a6000)
        assert latencies_consumer == sorted(latencies_consumer)
        assert ious == sorted(ious)

    def test_receipt_generation_and_gate_4_verdict(self, tmp_path):
        cfg_path = Path("training/configs/cfm_distill_qwen_image21.yaml")
        out_dir = tmp_path / "checkpoints"
        receipt_file = out_dir / "distillation_receipt.json"

        receipt = generate_cfm_distillation_receipt(
            config_path=cfg_path,
            output_dir=out_dir,
            receipt_path=receipt_file,
            selected_steps=6,
        )

        assert receipt_file.exists()
        assert receipt["phase"] == 4
        assert receipt["model"]["tier"] == "Tier.POLISH_FAST"
        assert receipt["distillation_config"]["huber_c"] == 0.001
        assert receipt["distillation_config"]["rank"] == 32
        assert receipt["verdict"]["gate_4_passed"] is True
        assert receipt["verdict"]["status"] == "VALIDATED"

        sel = receipt["selected_operating_point"]
        assert sel["steps"] == 6
        assert sel["meets_latency_gate_a6000"] is True
        assert sel["meets_latency_gate_consumer"] is True
        assert sel["meets_layout_gate"] is True
        assert sel["meets_ocr_gate"] is True
        assert sel["metrics"]["latency_seconds_a6000"] <= 1.0
        assert sel["metrics"]["latency_seconds_consumer"] <= 3.0

    def test_committed_checkpoint_receipt(self):
        receipt_path = Path("checkpoints/qwen_image21_cfm_fast/distillation_receipt.json")
        assert receipt_path.exists(), "Materialized distillation_receipt.json must exist"
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert data["phase"] == 4
        assert data["verdict"]["gate_4_passed"] is True
        assert data["verdict"]["status"] == "VALIDATED"


class TestPhase4RuntimeTierAndFlowIntegration:
    def test_model_registry_polish_fast_specs(self):
        # Default registry
        spec = REGISTRY[Tier.POLISH_FAST]
        assert spec.tier == Tier.POLISH_FAST
        assert spec.vram_gb == 10.0

        # Low-VRAM registry
        low_spec = LOW_VRAM_REGISTRY[Tier.POLISH_FAST]
        assert low_spec.vram_gb == 4.5
        assert low_spec.ram_gb == 6.0

        # High-VRAM registry
        high_spec = HIGH_VRAM_REGISTRY[Tier.POLISH_FAST]
        assert high_spec.vram_gb == 10.0
        assert high_spec.always_resident is True

    @pytest.mark.asyncio
    async def test_fast_preview_flow_execution(self, tmp_path):
        store = DesignStateStore(db_path=tmp_path / "sessions.db")
        state = store.create()
        # Make state finalize-eligible
        state.stage = SessionStage.SKETCHING
        state.layout.layout_ref = "layout://mock/spec"
        state.layout.wireframe_ref = "wireframe://mock/svg"
        rev = state.revision
        state.touch()
        store.save(state, rev)

        # Build mock orchestrator
        orchestrator = SwapOrchestrator(
            backend_factory=lambda spec: MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0),
            envelope_gb=24.0,
        )
        await orchestrator.start()
        try:
            res_state = await flows.fast_preview(
                orchestrator=orchestrator,
                store=store,
                session_id=state.session_id,
                prompt="A sleek modern fintech dashboard",
            )

            assert res_state.stage == SessionStage.FINALIZED
            assert res_state.preview_output is not None
            assert res_state.preview_output.step_count == 6
            assert res_state.preview_output.renderer_used == "qwen_image_2_1"
            assert res_state.finalize_output.image_ref is not None
        finally:
            await orchestrator.shutdown()
            store.close()
