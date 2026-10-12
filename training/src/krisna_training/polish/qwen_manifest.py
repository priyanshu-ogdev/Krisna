"""Versioned metadata contract for Qwen-Image-2.1 image-edit LoRAs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-2.1"
PIPELINE = "QwenImage21Pipeline"
SCHEMA_VERSION = 1

def build_manifest(*, adapter_path: str, resolution: int = 1024,
                   inference_steps: int = 40, rank: int | None = None,
                   lora_alpha: int | None = None,
                   training_script: str = "train_dreambooth_lora_qwenimage21_img2img.py",
                   diffusers_version: str | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "base_model": MODEL_ID,
            "pipeline": PIPELINE, "conditioning": "image",
            "training_kind": "dreambooth_lora_img2img",
            "adapter_path": str(Path(adapter_path)), "resolution": resolution,
            "inference_steps": inference_steps, "rank": rank,
            "lora_alpha": lora_alpha, "training_script": training_script,
            "diffusers_version": diffusers_version}

def write_manifest(adapter_path: str, **kwargs: Any) -> Path:
    path = Path(adapter_path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest(adapter_path=adapter_path, **kwargs), indent=2) + "\n")
    return manifest
