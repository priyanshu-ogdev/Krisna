"""Consistency and contract verification for active Qwen-Image-2.1 Polish training.

Validates:
1. Training LoRA config invariants and hyperparameter contracts.
2. Training/serving boundary manifests, receipts, and split counts.
3. Pair dataset manifest preparation and strict split preservation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

yaml = pytest.importorskip("yaml")

from krisna_training.polish.prepare_dataset import prepare_qwen_pairs_from_manifest
from krisna_training.polish.qwen_manifest import (
    BASE_REVISION,
    MODEL_ID,
    QwenManifestError,
    build_manifest,
    validate_manifest,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"


def _load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Qwen-Image-2.1 LoRA Configuration Invariants
# ---------------------------------------------------------------------------

def test_active_qwen_lora_config_sets_contract_explicitly() -> None:
    cfg = _load_config("polish_qwen_image21_lora.yaml")
    assert cfg["pretrained_model_name_or_path"] == "Qwen/Qwen-Image-2.1"
    assert cfg["rank"] > 0
    assert cfg["lora_alpha"] > 0
    assert cfg["cond_image_column"] == "condition_image"
    assert cfg["image_column"] == "image"
    assert cfg["caption_column"] == "caption"


def test_active_qwen_lora_config_has_observable_validation_settings() -> None:
    cfg = _load_config("polish_qwen_image21_lora.yaml")
    assert cfg["val_fraction"] is None
    assert cfg["val_every"] is None
    assert cfg["max_train_steps"] > cfg["lr_warmup_steps"]


# ---------------------------------------------------------------------------
# Training / Serving Boundary Manifest Contracts
# ---------------------------------------------------------------------------

def test_qwen_manifest_records_training_serving_boundary() -> None:
    manifest = build_manifest(
        adapter_path="checkpoints/qwen",
        base_revision=BASE_REVISION,
        dataset_manifest_sha256="b" * 64,
        split_counts={"train": 10, "validation": 2, "test": 2},
        lora_target_modules=["transformer.attn.to_q"],
        rank=32,
        trainable_parameter_count=100,
        frozen_parameter_count=1000,
        seed=7,
    )

    assert manifest["base_model"] == MODEL_ID
    assert manifest["pipeline"] == "QwenImage21Pipeline"
    assert validate_manifest(manifest)["split_counts"]["train"] == 10


def test_strict_qwen_manifest_rejects_missing_receipt() -> None:
    manifest = build_manifest(adapter_path="checkpoints/qwen", base_revision=BASE_REVISION)
    with pytest.raises(QwenManifestError, match="dataset_manifest_sha256"):
        validate_manifest(manifest)


def test_strict_qwen_manifest_rejects_unpinned_base() -> None:
    manifest = build_manifest(
        adapter_path="checkpoints/qwen",
        base_revision="a" * 40,
        dataset_manifest_sha256="b" * 64,
        split_counts={"train": 1},
        lora_target_modules=["to_q"],
    )
    with pytest.raises(QwenManifestError, match="base_revision"):
        validate_manifest(manifest)


def test_non_strict_validation_supports_legacy_operator_inspection() -> None:
    manifest = {
        "base_model": MODEL_ID,
        "pipeline": "QwenImagePipeline",
        "conditioning": "image",
    }
    assert validate_manifest(manifest, strict=False)["base_model"] == MODEL_ID


# ---------------------------------------------------------------------------
# Qwen Pair Manifest Preparation and Split Integrity
# ---------------------------------------------------------------------------

def test_prepare_qwen_pairs_preserves_receipts_and_splits(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    conditions = root / "conditions"
    targets = root / "targets"
    conditions.mkdir(parents=True)
    targets.mkdir(parents=True)
    condition = conditions / "pair.png"
    target = targets / "pair.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(condition)
    Image.new("RGB", (16, 16), (240, 240, 240)).save(target)
    row = {
        "record_id": "pair-1",
        "condition_image": "conditions/pair.png",
        "target_image": "targets/pair.png",
        "caption": "Refine the UI",
        "split": "train",
        "source_dataset": "test-ui",
        "source_revision": "a" * 40,
        "source_provenance": {"url": "https://example.test/ui"},
        "condition_sha256": _sha(condition),
        "target_sha256": _sha(target),
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = prepare_qwen_pairs_from_manifest(manifest, tmp_path / "prepared")
    metadata = json.loads((output / "metadata.jsonl").read_text().strip())
    receipt = json.loads((output / "dataset_receipt.json").read_text())
    assert metadata["pair_id"] == "pair-1"
    assert metadata["split"] == "train"
    assert metadata["source_provenance"]["url"].startswith("https://")
    assert receipt["split_counts"] == {"train": 1, "validation": 0, "test": 0}


def test_prepare_qwen_pairs_rejects_unassigned_split(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps({"record_id": "p", "split": "unassigned"}) + "\n")
    with pytest.raises(ValueError, match="split assignment"):
        prepare_qwen_pairs_from_manifest(manifest, tmp_path / "prepared")
