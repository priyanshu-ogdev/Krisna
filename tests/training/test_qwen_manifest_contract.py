from __future__ import annotations

import pytest

from krisna_training.polish.qwen_manifest import (
    BASE_REVISION,
    MODEL_ID,
    QwenManifestError,
    build_manifest,
    validate_manifest,
)


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
