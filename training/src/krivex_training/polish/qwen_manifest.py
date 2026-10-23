"""Versioned metadata contract for Qwen-Image-2.1 image-edit LoRAs.

The adapter manifest is the synchronization boundary between training and
serving. It records the exact base/pipeline contract and the data/run receipt;
directory names and launcher arguments are not sufficient provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-2.1"
PIPELINE = "QwenImage21Pipeline"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
SCHEMA_VERSION = 2


class QwenManifestError(ValueError):
    """Raised when a Qwen adapter manifest cannot be trusted by serving."""

def build_manifest(*, adapter_path: str, resolution: int = 1024,
                   inference_steps: int = 40, rank: int | None = None,
                   lora_alpha: int | None = None,
                   adaptation: str = "dora",
                   use_dora: bool = True,
                   training_script: str = "train_dreambooth_lora_qwenimage21_img2img.py",
                   diffusers_version: str | None = None,
                   base_revision: str | None = None,
                   dataset_manifest_sha256: str | None = None,
                   split_counts: dict[str, int] | None = None,
                   lora_target_modules: list[str] | None = None,
                   trainable_parameter_count: int | None = None,
                   frozen_parameter_count: int | None = None,
                   renderer_version: str = "krivex.renderer.v2",
                   prompt_template_version: str = "krivex.polish_prompt.v2",
                   seed: int | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "base_model": MODEL_ID,
            "pipeline": PIPELINE, "conditioning": "image",
            "training_kind": "dreambooth_lora_img2img",
            "adaptation": adaptation,
            "use_dora": use_dora,
            "adapter_path": str(Path(adapter_path)), "resolution": resolution,
            "inference_steps": inference_steps, "rank": rank,
            "lora_alpha": lora_alpha, "training_script": training_script,
            "diffusers_version": diffusers_version,
            "base_revision": base_revision,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_counts": split_counts or {},
            "lora_target_modules": lora_target_modules or [],
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": frozen_parameter_count,
            "renderer_version": renderer_version,
            "prompt_template_version": prompt_template_version,
            "seed": seed}


def validate_manifest(manifest: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Validate and return an adapter manifest without loading model weights."""
    if not isinstance(manifest, dict):
        raise QwenManifestError("manifest must be a JSON object")
    if manifest.get("base_model") != MODEL_ID:
        raise QwenManifestError(f"expected base_model {MODEL_ID!r}")
    if manifest.get("conditioning") != "image":
        raise QwenManifestError("Qwen adapter must declare image conditioning")
    if manifest.get("pipeline") not in {PIPELINE, "QwenImagePipeline"}:
        raise QwenManifestError("unsupported Qwen pipeline class")
    if strict:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise QwenManifestError(f"expected schema_version {SCHEMA_VERSION}")
        if manifest.get("base_revision") != BASE_REVISION:
            raise QwenManifestError(f"expected base_revision {BASE_REVISION}")
        if not manifest.get("dataset_manifest_sha256"):
            raise QwenManifestError("dataset_manifest_sha256 is required in strict mode")
        if not isinstance(manifest["dataset_manifest_sha256"], str) or len(manifest["dataset_manifest_sha256"]) != 64:
            raise QwenManifestError("dataset_manifest_sha256 must be a SHA-256 hex digest")
        counts = manifest.get("split_counts")
        if not isinstance(counts, dict) or not all(isinstance(v, int) and v >= 0 for v in counts.values()):
            raise QwenManifestError("split_counts must contain non-negative integers")
        if counts.get("train", 0) <= 0:
            raise QwenManifestError("split_counts.train must be positive")
        if not manifest.get("lora_target_modules"):
            raise QwenManifestError("lora_target_modules is required in strict mode")
        if not isinstance(manifest.get("rank"), int) or manifest["rank"] <= 0:
            raise QwenManifestError("rank must be a positive integer in strict mode")
        rv = manifest.get("renderer_version")
        if rv is not None and not (rv == "krivex.renderer.v2" or str(rv).startswith("krivex.renderer.v2.")):
            raise QwenManifestError(f"renderer_version mismatch: {manifest.get('renderer_version')}")
        if manifest.get("prompt_template_version") not in (None, "krivex.polish_prompt.v2"):
            raise QwenManifestError(f"prompt_template_version mismatch: {manifest.get('prompt_template_version')}")
    return manifest

def write_manifest(adapter_path: str, **kwargs: Any) -> Path:
    dataset_dir = kwargs.pop("dataset_dir", None)
    if not dataset_dir:
        raise QwenManifestError("dataset_dir is required so the training receipt is audited")
    base_revision = kwargs.get("base_revision")
    if base_revision != BASE_REVISION:
        raise QwenManifestError(f"expected base_revision {BASE_REVISION}")
    receipt_path = Path(dataset_dir) / "dataset_receipt.json"
    if not receipt_path.is_file():
        raise QwenManifestError(f"dataset receipt is missing: {receipt_path}")
    try:
        from .qwen_trainer_contract import audit_prepared_dataset
        audited = audit_prepared_dataset(dataset_dir)
        receipt = audited["receipt"]
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise QwenManifestError(f"dataset receipt audit failed: {exc}") from exc
    if kwargs.get("dataset_manifest_sha256") != receipt.get("manifest_sha256"):
        raise QwenManifestError("dataset_manifest_sha256 does not match dataset receipt")
    if kwargs.get("split_counts") != receipt.get("split_counts"):
        raise QwenManifestError("split_counts do not match dataset receipt")
    path = Path(adapter_path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    payload = build_manifest(adapter_path=adapter_path, **kwargs)
    validate_manifest(payload, strict=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest
