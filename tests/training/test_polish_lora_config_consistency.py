"""Consistency checks for the active Qwen-Image-2.1 LoRA contract.

Historical Z-Image/DPO configuration variants are retired and are intentionally
not treated as evidence for the current training path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def test_active_qwen_lora_config_sets_contract_explicitly() -> None:
    cfg = _load("polish_qwen_image21_lora.yaml")
    assert cfg["pretrained_model_name_or_path"] == "Qwen/Qwen-Image-2.1"
    assert cfg["rank"] > 0
    assert cfg["lora_alpha"] > 0
    assert cfg["cond_image_column"] == "condition_image"
    assert cfg["image_column"] == "image"
    assert cfg["caption_column"] == "caption"


def test_active_qwen_lora_config_has_observable_validation_settings() -> None:
    cfg = _load("polish_qwen_image21_lora.yaml")
    assert cfg["val_fraction"] is None
    assert cfg["val_every"] is None
    assert cfg["max_train_steps"] > cfg["lr_warmup_steps"]
