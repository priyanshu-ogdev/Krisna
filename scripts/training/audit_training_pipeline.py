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
    backend = ROOT / "inference/src/krivex_inference/backends/polish_backend_qwen.py"
    planner_download = ROOT / "scripts/inference/download_planner_models.py"

    pair_dir = ROOT / "data/polish_qwen_image21_pairs"
    data_roots = [ROOT / "data_krivex", ROOT / "custom_data_root"]

    qwen_trainer = ROOT / "third_party/diffusers/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"
    report: dict[str, Any] = {
        "repository": str(ROOT),
        "model_ids": {
            "polish_default_and_quality": _regex(
                config, r'^pretrained_model_name_or_path:\s*["\']([^"\']+)', "Qwen/Qwen-Image-2.1"
            ),
            "polish_backend_default": _regex(
                backend, r'KRIVEX_POLISH_MODEL_ID",\s*"([^"]+)"', "Qwen/Qwen-Image-2.1"
            ),
            "planner_gguf_repository": _regex(
                planner_download, r'KRIVEX_PLANNER_GGUF_REPO",\s*"([^"]+)"', "unsloth/Qwen3.5-9B-GGUF"
            ),
        },
        "trainability": {
            "planner": "frozen (Qwen3.5-9B canonical + UICrit RAG, PRD §6 no-RLHF)",
            "layout": "deterministic baseline (krivex.layout.v1 Layout IR); LayoutFlow/MaskGIT retired",
            "polish_default": "intended LoRA/DoRA; trainer contract pinned to Qwen-Image-2.1",
            "polish_quality": "intended frozen; verify no adapter is injected",
            "polish_dpo": "active Flow-Matching DPO preference alignment (Wallace et al. 2024; Kim et al. 2024)",
            "polish_cfm": "active Consistency Flow Matching distillation (Luo et al. 2024; Song et al. 2023)",
        },
        "source_entry_points": {
            "qwen_pair_builder": _exists("training/src/krivex_training/polish/prepare_dataset.py"),
            "qwen_trainer_contract": _exists("training/src/krivex_training/polish/qwen_trainer_contract.py"),
            "external_qwen_trainer": qwen_trainer.exists(),
            "layout_model_primitive": False,
            "layout_trainer": False,
            "sketch_trainer": False,
            "dpo_loss_source": _exists("training/src/krivex_training/polish/dpo_loss.py"),
            "dpo_trainer": _exists("training/src/krivex_training/polish/train_dpo.py"),
            "cfm_loss_source": _exists("training/src/krivex_training/polish/cfm_loss.py"),
            "ablation_runner": _exists("training/src/krivex_training/polish/ablation_runner.py"),
        },
        "launchers": {
            "qwen_default_bash": _exists("scripts/training/train_polish_default_lora.sh"),
            "qwen_dpo_bash": _exists("scripts/training/train_polish_dpo.sh"),
            "qwen_cfm_bash": _exists("scripts/training/train_polish_fast_preview.sh"),
            "windows_dispatcher": _exists("scripts/training/train_all.ps1"),
            "missing_sketch_stage1_bash": not _exists("scripts/training/train_sketch_stage1.sh"),
            "missing_sketch_stage2_bash": not _exists("scripts/training/train_sketch_stage2.sh"),
        },
        "realized_data": {
            "qwen_pair_directory_exists": pair_dir.is_dir(),
            "qwen_pair_files": _file_count(pair_dir),
            "qwen_metadata_rows": _metadata_rows(pair_dir / "metadata.jsonl"),
            "data_roots": [],
            "preference_db": _preference_count(ROOT / "krivex_preference_pairs.db"),
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
        "canonical_architecture": "Single-Trainable-Tier (PRD §6 & FINAL_ARCHITECTURE_VERDICT.md)",
        "active_trainable_models": [
            "Qwen/Qwen-Image-2.1 Supervised LoRA/DoRA (polish-default: Primary Production Target)",
        ],
        "deferred_research_extensions": [
            "Qwen/Qwen-Image-2.1 Flow-Matching DPO (polish-dpo: Deferred post-convergence alignment)",
            "Qwen/Qwen-Image-2.1 Consistency Flow Matching Distillation (polish-cfm: Experimental fast preview)",
        ],
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

    print("Krivex training pipeline audit (read-only)")
    print(f"Repository: {report['repository']}")
    print("Primary trainable tier (PRD §6 Single-Trainable-Tier):")
    for m in report['verdict']['active_trainable_models']:
        print(f"  - {m}")
    print("Deferred / Experimental research extensions:")
    for m in report['verdict'].get('deferred_research_extensions', []):
        print(f"  - {m}")
    print(f"Qwen trainer contract present: {report['source_entry_points']['qwen_trainer_contract']}")
    print(f"DPO trainer present: {report['source_entry_points']['dpo_trainer']}")
    print(f"CFM loss source present: {report['source_entry_points']['cfm_loss_source']}")
    print(f"Qwen pair directory present: {report['realized_data']['qwen_pair_directory_exists']}")
    print(f"Qwen metadata rows: {report['realized_data']['qwen_metadata_rows']}")
    print(f"Preference DB rows: {report['realized_data']['preference_db']['total']}")
    print(f"Sketch trainer present: {report['source_entry_points']['sketch_trainer']}")
    print("Planner trained: False (frozen)")
    print("Learned Sketch trained: False (deterministic Layout IR baseline)")
    print("Note: this audit does not install, download, or launch training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
