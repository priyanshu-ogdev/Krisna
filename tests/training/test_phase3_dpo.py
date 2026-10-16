"""Tests for Phase 3: Continuous Flow Matching DPO, Curriculum Scheduling, and Run Receipts.
Theoretical Grounding:
- Wallace et al., IEEE/CVF CVPR 2024
- Kim et al., NeurIPS 2024
- Croitoru et al., IEEE/CVF CVPR 2025
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from krisna_training.polish.dpo_loss import (
    FlowMatchingDPOLossOutput,
    flow_matching_dpo_loss,
    flow_matching_velocity_target,
    noise_latent_at_timestep,
)
from scripts.training.compare_dpo_betas import compute_dataset_stats, generate_phase3_receipt
from scripts.training.dpo_lora_autodetect import resolve_lora_adapter_path


def _rand_latent(batch=2, channels=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, h, w, generator=g)


class TestPhase3DPOLoss:
    def test_curriculum_dpo_modulation(self):
        """Croitoru et al. (CVPR 2025): curriculum margin scheduling attenuates loss
        on examples below the curriculum threshold."""
        target_chosen = _rand_latent(seed=101)
        target_rejected = _rand_latent(seed=102)
        ref_pred = _rand_latent(seed=103)
        policy_chosen = _rand_latent(seed=104)
        policy_rejected = _rand_latent(seed=105)

        # Baseline loss without curriculum
        out_standard = flow_matching_dpo_loss(
            policy_v_chosen=policy_chosen,
            policy_v_rejected=policy_rejected,
            ref_v_chosen=ref_pred,
            ref_v_rejected=ref_pred,
            target_v_chosen=target_chosen,
            target_v_rejected=target_rejected,
            beta=2000.0,
            curriculum_margin=None,
        )

        # Loss with curriculum modulation on easy pair (pair_margin < curriculum_margin)
        out_curriculum_easy = flow_matching_dpo_loss(
            policy_v_chosen=policy_chosen,
            policy_v_rejected=policy_rejected,
            ref_v_chosen=ref_pred,
            ref_v_rejected=ref_pred,
            target_v_chosen=target_chosen,
            target_v_rejected=target_rejected,
            beta=2000.0,
            curriculum_margin=1.0,
            pair_margins=0.2,
        )

        assert out_curriculum_easy.dpo_term.item() < out_standard.dpo_term.item()
        assert out_curriculum_easy.loss.item() < out_standard.loss.item()

    def test_all_output_fields_populated(self):
        target = _rand_latent(seed=201)
        ref = _rand_latent(seed=202)
        policy = _rand_latent(seed=203)

        out = flow_matching_dpo_loss(
            policy_v_chosen=policy,
            policy_v_rejected=policy,
            ref_v_chosen=ref,
            ref_v_rejected=ref,
            target_v_chosen=target,
            target_v_rejected=target,
            beta=2000.0,
            fm_anchor_weight=0.1,
        )
        assert isinstance(out, FlowMatchingDPOLossOutput)
        assert out.loss is not None
        assert out.dpo_term is not None
        assert out.fm_anchor_term is not None
        assert out.implicit_reward_margin is not None
        assert out.chosen_policy_err is not None
        assert out.rejected_policy_err is not None


class TestPhase3ConfigAndReceipt:
    def test_qwen_image21_dpo_yaml_validity(self):
        cfg_path = Path("training/configs/polish_qwen_image21_dpo.yaml")
        assert cfg_path.exists(), "polish_qwen_image21_dpo.yaml must exist"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        assert data["pretrained_model_name_or_path"] == "Qwen/Qwen-Image-2.1"
        assert data["lora_rank"] == 64
        assert data["lora_alpha"] == 32
        assert data["use_dora"] is True
        assert data["beta"] == 2000.0
        assert data["fm_anchor_weight"] == 0.1
        assert "ff.net.0.proj" in data["lora_target_modules"]
        assert "to_q" in data["lora_target_modules"]
        assert data["resolution"] == 1024

    def test_preference_db_gate_and_receipt_generation(self, tmp_path):
        db_path = Path("krisna_preference_pairs.db")
        assert db_path.exists(), "krisna_preference_pairs.db must exist"

        stats = compute_dataset_stats(db_path)
        assert stats["total"] >= 2000, f"Phase 3 Gate: Expected >= 2000 pairs, found {stats['total']}"
        assert stats["val"] > 0, "Validation split must contain heldout pairs"

        receipt_file = tmp_path / "training_receipt.json"
        receipt = generate_phase3_receipt(
            db_path=db_path,
            output_dir=tmp_path,
            receipt_path=receipt_file,
            betas=[500.0, 2000.0, 5000.0],
            anchor_weight=0.1,
            selected_beta=2000.0,
        )

        assert receipt_file.exists()
        assert receipt["phase"] == 3
        assert receipt["dataset"]["gate_status"] == "PASS"
        assert receipt["verdict"]["gate_3_passed"] is True
        assert "beta_2000" in receipt["beta_sweep_evaluation"]
        assert receipt["selected_metrics"]["reward_accuracy"] > 0.85

    def test_dpo_lora_autodetect_integration(self, tmp_path):
        cfg_path = Path("training/configs/polish_qwen_image21_dpo.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        fake_base = tmp_path / "polish_default_lora"
        fake_base.mkdir()
        (fake_base / "adapter_model.safetensors").write_bytes(b"mock")

        updated_cfg, msg = resolve_lora_adapter_path(cfg, fake_base)
        assert updated_cfg["lora-adapter-path"] == str(fake_base)
        assert "auto-continuing" in msg
