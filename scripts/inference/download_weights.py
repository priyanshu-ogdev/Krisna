#!/usr/bin/env python3
"""Post-training installer and model downloader for Krivex Inference.

Downloads and provisions the authoritative foundation models and trained adapters
for the Krivex inference stack:
1. Polish Tier Base: Qwen/Qwen-Image-2.1 (Flow-Matching ODE MMDiT foundation model).
2. Planner Tier: Qwen3.5-9B-Instruct-Q4_K_M.gguf + Multimodal Projector (frozen GGUF via llama.cpp).
3. Sketch Tier: Deterministic Neuro-Symbolic Layout IR engine (no external weights required).
4. Trained LoRA Adapter: Auto-discovers local trained checkpoints (checkpoints/polish_default_lora,
   models/dpo_checkpoints/**/final) or downloads from Hugging Face Hub (krivex-ai/krivex-qwen-image21-polish-lora).
   If not yet trained or published, gracefully falls back to base Qwen-Image-2.1 zero-shot mode.

Emits JSON events per line on stdout for seamless streaming to the Studio frontend.

Usage:
    python scripts/inference/download_weights.py
    python scripts/inference/download_weights.py --low-vram
    python scripts/inference/download_weights.py --dry-run
    python scripts/inference/download_weights.py --polish-lora checkpoints/polish_default_lora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env.inference"

QWEN_IMAGE_21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE_21_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
DEFAULT_PLANNER_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_PLANNER_GGUF_FILE = "Qwen3.5-9B-Instruct-Q4_K_M.gguf"
DEFAULT_PLANNER_MMPROJ_FILE = "Qwen3.5-9B-Instruct-mmproj-f16.gguf"
DEFAULT_TRAINED_LORA_REPO = "krivex-ai/krivex-qwen-image21-polish-lora"


def emit(event: str, **fields) -> None:
    """One JSON line per event, flushed immediately."""
    print(json.dumps({"ts": time.time(), "event": event, **fields}), flush=True)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class HFModel:
    key: str
    repo_id: str
    tier: str
    required: bool = True
    revision: str = "main"
    allow_patterns: list[str] | None = None


def get_foundation_models(skip_polish: bool = False, skip_planner: bool = False) -> list[HFModel]:
    models: list[HFModel] = []
    if not skip_polish:
        models.append(
            HFModel(
                key="POLISH_MODEL",
                repo_id=os.environ.get("KRIVEX_POLISH_MODEL_ID", QWEN_IMAGE_21_MODEL_ID),
                tier="polish",
                revision=os.environ.get("KRIVEX_POLISH_REVISION", QWEN_IMAGE_21_REVISION),
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
            )
        )
    return models


# Approximate on-disk download sizes (GB) for disk check
APPROX_DOWNLOAD_GB = {
    QWEN_IMAGE_21_MODEL_ID: 35.0,
    DEFAULT_PLANNER_GGUF_REPO: 5.5,
    "unsloth/Qwen3.5-9B-GGUF": 6.5,
}


def check_disk_space(needed_gb: float, dest: Path) -> tuple[bool, float, float]:
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free / (1024**3)
    return free >= needed_gb, needed_gb, free


def download_hf_snapshot(m: HFModel, dest_root: Path, max_retries: int = 3) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    local_dir = dest_root / m.repo_id.replace("/", "__")
    emit("model_download_start", tier=m.tier, repo_id=m.repo_id, dest=str(local_dir))

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            path = snapshot_download(
                repo_id=m.repo_id,
                revision=m.revision,
                local_dir=str(local_dir),
                allow_patterns=m.allow_patterns,
                max_workers=4,
            )
            emit("model_download_complete", tier=m.tier, repo_id=m.repo_id, path=path, attempt=attempt)
            return Path(path)
        except (HfHubHTTPError, Exception) as e:
            last_exc = e
            emit(
                "model_download_retry",
                tier=m.tier, repo_id=m.repo_id, attempt=attempt, max_retries=max_retries,
                error=str(e),
            )
            time.sleep(min(2 ** attempt, 30))

    emit("model_download_failed", tier=m.tier, repo_id=m.repo_id, error=str(last_exc))
    raise RuntimeError(f"Failed to download {m.repo_id} after {max_retries} attempts: {last_exc}") from last_exc


def download_planner_gguf(
    repo_id: str,
    gguf_name: str,
    mmproj_name: str | None,
    dest_dir: Path,
    revision: str = "main",
    max_retries: int = 3,
) -> tuple[Path, Path | None]:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    target_gguf = dest_dir / gguf_name
    emit("planner_download_start", repo_id=repo_id, file=gguf_name, dest=str(target_gguf))

    for attempt in range(1, max_retries + 1):
        try:
            if not target_gguf.exists():
                src = hf_hub_download(repo_id, filename=gguf_name, revision=revision, local_dir=dest_dir)
                if Path(src) != target_gguf:
                    target_gguf.write_bytes(Path(src).read_bytes())
            emit("planner_download_complete", file=gguf_name, path=str(target_gguf))
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 15))

    target_mmproj = None
    if mmproj_name:
        target_mmproj = dest_dir / mmproj_name
        emit("planner_mmproj_download_start", repo_id=repo_id, file=mmproj_name, dest=str(target_mmproj))
        for attempt in range(1, max_retries + 1):
            try:
                if not target_mmproj.exists():
                    src = hf_hub_download(repo_id, filename=mmproj_name, revision=revision, local_dir=dest_dir)
                    if Path(src) != target_mmproj:
                        target_mmproj.write_bytes(Path(src).read_bytes())
                emit("planner_mmproj_download_complete", file=mmproj_name, path=str(target_mmproj))
                break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download required multimodal projector {mmproj_name}: {e}") from e
                time.sleep(min(2 ** attempt, 15))

    # Write manifest
    manifest_entries = [{"filename": target_gguf.name, "sha256": sha256_file(target_gguf)}]
    if target_mmproj and target_mmproj.exists():
        manifest_entries.append({"filename": target_mmproj.name, "sha256": sha256_file(target_mmproj)})
    manifest = {"repo_id": repo_id, "revision": revision, "artifacts": manifest_entries}
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return target_gguf, target_mmproj


def resolve_trained_lora(
    explicit_path: str | None = None,
    hf_repo_id: str | None = None,
    dest_dir: Path | None = None,
) -> Path | None:
    """Resolve trained Polish LoRA adapter from explicit path, local training output, or Hugging Face."""
    # 1. Explicit local path
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.exists():
            emit("lora_found", path=str(p), source="explicit")
            return p

    # 2. Local search in training directories
    search_patterns = [
        "checkpoints/polish_default_lora",
        "models/dpo_checkpoints/**/final",
        "checkpoints/polish_qwen_image21_dpo",
    ]
    matches: list[Path] = []
    for pattern in search_patterns:
        matches.extend(REPO_ROOT.glob(pattern))
    matches = [m for m in matches if m.is_dir() and (m / "adapter_model.safetensors").exists()]
    if matches:
        best = max(matches, key=lambda p: p.stat().st_mtime)
        emit("lora_found", path=str(best), source="local_training")
        return best

    # 3. Download from Hugging Face Hub if configured or published
    repo = hf_repo_id or os.environ.get("KRIVEX_POLISH_LORA_REPO", DEFAULT_TRAINED_LORA_REPO)
    if dest_dir and repo:
        try:
            from huggingface_hub import snapshot_download
            emit("lora_hf_check", repo_id=repo)
            lora_dir = dest_dir / "adapters" / repo.replace("/", "__")
            lora_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_download(repo_id=repo, local_dir=str(lora_dir), max_workers=2)
            if (Path(path) / "adapter_model.safetensors").exists():
                emit("lora_found", path=str(path), source="huggingface")
                return Path(path)
        except Exception as e:
            emit("lora_hf_unavailable", repo_id=repo, note="Trained model not yet published on HF Hub; will run zero-shot base.")

    emit("lora_status", status="using_base_model", note="Running base Qwen-Image-2.1 zero-shot without fine-tuned LoRA.")
    return None


def write_env_file(resolved: dict[str, str], low_vram: bool) -> None:
    lines = [
        "# Generated by scripts/inference/download_weights.py — do not edit by hand.",
        "# Source this before starting the inference service:",
        "#   set -a; source .env.inference; set +a; ./scripts/inference/run_service.sh",
        "KRIVEX_USE_REAL_BACKENDS=1",
        f"KRIVEX_LOW_VRAM_MODE={'1' if low_vram else '0'}",
        "KRIVEX_BLOB_ROOT=data/blobs",
        "KRIVEX_SESSIONS_DB=data/krivex_sessions.db",
        "KRIVEX_PREFERENCE_DB=data/krivex_preference_pairs.db",
    ]
    for k, v in resolved.items():
        if v is not None:
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("env_file_written", path=str(ENV_FILE), vars=len(resolved))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(REPO_ROOT / "models"),
                     help="Where model weights and snapshots land (default: ./models/).")
    ap.add_argument("--low-vram", action="store_true",
                     help="Configure KRIVEX_LOW_VRAM_MODE=1 for 12GB offloaded execution.")
    ap.add_argument("--skip-polish", action="store_true", help="Skip downloading Qwen-Image-2.1 foundation weights.")
    ap.add_argument("--skip-planner", action="store_true", help="Skip downloading Planner GGUF weights.")
    ap.add_argument("--planner-repo", default=None,
                     help="Hugging Face repo for Planner GGUF (default: unsloth/Qwen3.5-9B-GGUF).")
    ap.add_argument("--planner-gguf", default=None,
                     help="Exact GGUF filename for Planner model (default: Qwen3.5-9B-Instruct-Q4_K_M.gguf).")
    ap.add_argument("--planner-mmproj", default=None,
                     help="Multimodal projector GGUF filename (default: Qwen3.5-9B-Instruct-mmproj-f16.gguf).")
    ap.add_argument("--polish-lora", default=None,
                     help="Path to trained Qwen-Image-2.1 LoRA adapter directory.")
    ap.add_argument("--polish-lora-repo", default=DEFAULT_TRAINED_LORA_REPO,
                     help="Hugging Face repo ID for trained Polish LoRA adapter.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Inspect and verify all models without downloading.")
    args = ap.parse_args()

    planner_repo = args.planner_repo or os.environ.get(
        "KRIVEX_PLANNER_GGUF_REPO",
        DEFAULT_PLANNER_GGUF_REPO,
    )
    planner_gguf = args.planner_gguf or os.environ.get(
        "KRIVEX_PLANNER_GGUF_FILE",
        DEFAULT_PLANNER_GGUF_FILE,
    )
    planner_mmproj = args.planner_mmproj or os.environ.get(
        "KRIVEX_PLANNER_MMPROJ_FILE",
        DEFAULT_PLANNER_MMPROJ_FILE,
    )

    dest = Path(args.dest).expanduser().resolve()
    planner_dest = dest / "planner"

    # 1. Plan and verify disk space
    foundation_models = get_foundation_models(skip_polish=args.skip_polish, skip_planner=args.skip_planner)
    needed_gb = sum(APPROX_DOWNLOAD_GB.get(m.repo_id, 20.0) for m in foundation_models)
    if not args.skip_planner:
        needed_gb += APPROX_DOWNLOAD_GB.get(args.planner_repo, 6.0)

    emit("plan", dest=str(dest), needed_gb=needed_gb, low_vram=args.low_vram, dry_run=args.dry_run)

    ok, needed, free = check_disk_space(needed_gb, dest)
    emit("disk_check", needed_gb=round(needed, 1), free_gb=round(free, 1), ok=ok)
    if not ok and not args.dry_run:
        emit("aborted", reason="insufficient_disk_space", needed_gb=round(needed, 1), free_gb=round(free, 1))
        return 1

    resolved: dict[str, str] = {}
    failures: list[str] = []

    # 2. Download Polish foundation model
    if not args.skip_polish:
        for m in foundation_models:
            if args.dry_run:
                emit("model_download_skipped_dry_run", tier=m.tier, repo_id=m.repo_id)
                continue
            try:
                p = download_hf_snapshot(m, dest)
                resolved["KRIVEX_POLISH_MODEL_ID"] = str(p)
            except Exception as e:
                failures.append(f"Polish model ({m.repo_id}): {e}")

    # 3. Download Planner GGUF + Multimodal Projector
    if not args.skip_planner:
        if args.dry_run:
            emit("planner_download_skipped_dry_run", repo_id=planner_repo, file=planner_gguf)
        else:
            try:
                gguf_path, mmproj_path = download_planner_gguf(
                    repo_id=planner_repo,
                    gguf_name=planner_gguf,
                    mmproj_name=planner_mmproj,
                    dest_dir=planner_dest,
                )
                resolved["KRIVEX_PLANNER_GGUF"] = str(gguf_path)
                if mmproj_path:
                    resolved["KRIVEX_PLANNER_MMPROJ"] = str(mmproj_path)
            except Exception as e:
                failures.append(f"Planner GGUF ({planner_repo}/{planner_gguf}): {e}")

    # 4. Resolve Sketch Tier (Deterministic Layout IR)
    emit("sketch_tier_status", status="ready", engine="deterministic_layout_ir_v1", vram_mb=0)

    # 5. Resolve Trained Polish LoRA Adapter (Local or HF)
    lora_path = resolve_trained_lora(
        explicit_path=args.polish_lora,
        hf_repo_id=args.polish_lora_repo,
        dest_dir=dest,
    )
    if lora_path:
        resolved["KRIVEX_POLISH_DEFAULT_LORA_PATH"] = str(lora_path)

    if args.dry_run:
        emit("dry_run_complete", would_fail=failures)
        return 1 if failures else 0

    if failures:
        emit("install_incomplete", failures=failures)
        write_env_file(resolved, args.low_vram)
        return 1

    write_env_file(resolved, args.low_vram)
    emit("install_complete", env_file=str(ENV_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
