#!/usr/bin/env python3
"""Read-only audit of the training source tree and realized artifacts.

This intentionally does not import torch/diffusers, install packages, clone
repositories, download weights, or launch a trainer. It answers a narrower and
more useful question: what does this checkout claim to train, what source
entry points exist, and which data/checkpoint artifacts are actually present?

Usage:
    python scripts/training/audit_training_pipeline.py
    python scripts/training/audit_training_pipeline.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _exists(relative: str) -> bool:
    return (ROOT / relative).exists()


def _file_count(path: Path, pattern: str = "*") -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def _regex(path: Path, pattern: str, default: str | None = None) -> str | None:
    if not path.is_file():
        return default
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else default


def _preference_count(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.is_file(), "total": 0, "sources": {}}
    if not path.is_file():
        return result
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        result["total"] = int(connection.execute("SELECT count(*) FROM preference_pairs").fetchone()[0])
        rows = connection.execute(
            "SELECT source, count(*) FROM preference_pairs GROUP BY source ORDER BY source"
        ).fetchall()
        result["sources"] = {str(source): int(count) for source, count in rows}
        connection.close()
    except (OSError, sqlite3.Error) as error:
        result["error"] = str(error)
    return result


def collect() -> dict[str, Any]:
    config = ROOT / "training/configs/polish_qwen_image21_lora.yaml"
    backend = ROOT / "inference/src/krisna_inference/backends/polish_backend_qwen.py"
    planner_download = ROOT / "scripts/inference/download_planner_models.py"

    pair_dir = ROOT / "data/polish_qwen_image21_pairs"
    data_roots = [ROOT / "data_krisna", ROOT / "custom_data_root"]

    qwen_trainer = ROOT / "third_party/diffusers/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"
    report: dict[str, Any] = {
        "repository": str(ROOT),
        "model_ids": {
            "polish_default_and_quality": _regex(
                config, r'^pretrained_model_name_or_path:\s*["\']([^"\']+)', "Qwen/Qwen-Image-2.1"
            ),
            "polish_backend_default": _regex(
                backend, r'KRISNA_POLISH_MODEL_ID",\s*"([^"]+)"', "Qwen/Qwen-Image-2.1"
            ),
            "planner_gguf_repository": _regex(
                planner_download, r'KRISNA_PLANNER_GGUF_REPO",\s*"([^"]+)"', "ggml-org/Qwen3.5-9B-GGUF"
            ),
        },
        "trainability": {
            "planner": "frozen",
            "layout": "deterministic baseline; learned trainer not present",
            "polish_default": "intended LoRA; trainer contract pending validation",
            "polish_quality": "intended frozen; verify no adapter is injected",
            "dpo": "deferred historical path",
        },
        "source_entry_points": {
            "qwen_pair_builder": _exists("training/src/krisna_training/polish/prepare_dataset.py"),
            "qwen_local_trainer": _exists("training/src/krisna_training/polish/train_qwen_image21.py"),
            "external_qwen_trainer": qwen_trainer.exists(),
            "layout_model_primitive": _exists("training/src/krisna_training/layout/model.py"),
            "layout_trainer": _exists("training/src/krisna_training/layout/train.py"),
            "sketch_trainer": _exists("training/src/krisna_training/sketch/train.py"),
            "dpo_loss_source": _exists("training/src/krisna_training/polish/dpo_loss.py"),
        },
        "launchers": {
            "qwen_default_bash": _exists("scripts/training/train_polish_default_lora.sh"),
            "retired_dpo_bash": _exists("scripts/training/train_polish_dpo.sh"),
            "windows_dispatcher": _exists("scripts/training/train_all.ps1"),
            "missing_sketch_stage1_bash": not _exists("scripts/training/train_sketch_stage1.sh"),
            "missing_sketch_stage2_bash": not _exists("scripts/training/train_sketch_stage2.sh"),
        },
        "realized_data": {
            "qwen_pair_directory_exists": pair_dir.is_dir(),
            "qwen_pair_files": _file_count(pair_dir),
            "qwen_metadata_rows": _metadata_rows(pair_dir / "metadata.jsonl"),
            "data_roots": [],
            "preference_db": _preference_count(ROOT / "krisna_preference_pairs.db"),
        },
        "artifacts": {
            "models_files": _file_count(ROOT / "models", "*") if (ROOT / "models").is_dir() else 0,
            "qwen_manifest_exists": _exists("checkpoints/qwen_image21_lora/manifest.json"),
        },
    }

    for data_root in data_roots:
        report["realized_data"]["data_roots"].append(
            {
                "path": str(data_root),
                "exists": data_root.is_dir(),
                "model_data_files": _file_count(data_root / "model_data"),
                "preference_json_files": _file_count(data_root / "preference_pairs", "*.json"),
            }
        )

    report["verdict"] = {
        "active_trainable_model": "Qwen/Qwen-Image-2.1 LoRA (pending trainer validation)",
        "planner_trained": False,
        "learned_sketch_trained": False,
        "qwen_training_ready": bool(qwen_trainer.exists() and pair_dir.is_dir()),
        "warning": "Configured paths are not evidence of a completed training run.",
    }
    return report


def _metadata_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    report = collect()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Krisna training pipeline audit (read-only)")
    print(f"Repository: {report['repository']}")
    print(f"Active trainable model: {report['verdict']['active_trainable_model']}")
    print(f"Qwen trainer present: {report['source_entry_points']['external_qwen_trainer']}")
    print(f"Qwen pair directory present: {report['realized_data']['qwen_pair_directory_exists']}")
    print(f"Qwen metadata rows: {report['realized_data']['qwen_metadata_rows']}")
    print(f"Preference DB rows: {report['realized_data']['preference_db']['total']}")
    print(f"Sketch trainer present: {report['source_entry_points']['sketch_trainer']}")
    print(f"DPO loss source present: {report['source_entry_points']['dpo_loss_source']}")
    print("Planner trained: False (frozen)")
    print("Learned Sketch trained: False (deterministic Layout IR baseline)")
    print("Note: this audit does not install, download, or launch training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
