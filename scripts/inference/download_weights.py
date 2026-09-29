#!/usr/bin/env python3
"""Post-training installer for the inference layer.

Run this ONCE after training finishes (or any time you need to
(re)provision a machine) and BEFORE `KRISNA_USE_REAL_BACKENDS=1
./scripts/inference/run_service.sh`. It does two distinct jobs, kept
in one script because a real deployment always needs both together:

1. DOWNLOAD the four frozen, HF-hosted models real backends load
   (see inference/backends/factory.py for exactly how each is wired):
     - Planner   : Qwen/Qwen3.5-9B      (or Qwen/Qwen3.5-4B, --fast-planner)
     - Polish    : Tongyi-MAI/Z-Image-Turbo
     - Quality   : Qwen/Qwen-Image-Edit-2511   (skippable, --skip-quality —
                   it's the optional higher-VRAM tier with a fallback_tier)
     - Critic    : unsloth/gemma-4-31B-it-unsloth-bnb-4bit  (skippable,
                   --skip-critic — separate venv, on-demand feature)

2. LOCATE this project's own trained artifacts (Sketch tier checkpoint,
   Z-Image-Turbo LoRA) and verify they actually exist where the real
   backends expect to find them (KRISNA_SKETCH_CHECKPOINT,
   KRISNA_POLISH_DEFAULT_LORA_PATH) — these are NOT downloaded from HF,
   they only exist after a real training run on THIS repo.

Everything resolved is written to a single `.env.inference` file at the
monorepo root that `run_service.sh` sources, so "did I actually finish
installing" is one file to check, not five different places.

Usage:
    python scripts/inference/download_weights.py \\
        --sketch-checkpoint checkpoints/sketch_stage2_512/checkpoint_final.pt \\
        --polish-lora models/dpo_checkpoints/stage1_general/final

    python scripts/inference/download_weights.py --low-vram --skip-critic

    python scripts/inference/download_weights.py --dry-run

Progress is emitted as one JSON object per line on stdout — this is
deliberate, not an accident of logging config: it's what lets a thin
frontend (see inference-frontend/) tail this script's stdout and render
a real progress bar/log without parsing free-form text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env.inference"


def emit(event: str, **fields) -> None:
    """One JSON line per event, flushed immediately. This is the whole
    contract between this script and anything consuming its output
    (inference-frontend/server.js's SSE relay, a CI log, a human)."""
    print(json.dumps({"ts": time.time(), "event": event, **fields}), flush=True)


@dataclass
class HFModel:
    key: str                 # matches the KRISNA_* env var this feeds, minus the prefix
    repo_id: str
    tier: str
    required: bool = True
    revision: str = "main"
    allow_patterns: list[str] | None = None  # e.g. skip .bin when safetensors exist


def frozen_models(low_vram: bool, fast_planner: bool, skip_quality: bool, skip_critic: bool) -> list[HFModel]:
    return [
        HFModel(
            key="PLANNER_MODEL",
            repo_id="Qwen/Qwen3.5-4B" if fast_planner else "Qwen/Qwen3.5-9B",
            tier="planner",
        ),
        HFModel(
            key="POLISH_DEFAULT_MODEL",
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            tier="polish_default",
        ),
        HFModel(
            key="POLISH_QUALITY_MODEL",
            repo_id="Qwen/Qwen-Image-Edit-2511",
            tier="polish_quality",
            required=not skip_quality,
        ),
        HFModel(
            key="CRITIC_MODEL",
            repo_id="unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
            tier="critic",
            required=not skip_critic,
            # Pre-quantized NF4 checkpoint ships safetensors only; explicit
            # allow_patterns avoids pulling any stray non-safetensors shards
            # some mirrors of quantized repos carry as legacy artifacts.
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
        ),
    ]


# Rough, declared disk footprints (GB) — used only for the preflight
# free-space check below, deliberately conservative (rounded up from the
# NF4/quantized on-disk size, not the VRAM figure in model_registry.py,
# which is a different number: on-disk safetensors vs. loaded-tensor
# VRAM residency are not the same size).
APPROX_DOWNLOAD_GB = {
    "Qwen/Qwen3.5-9B": 10.0,
    "Qwen/Qwen3.5-4B": 5.0,
    "Tongyi-MAI/Z-Image-Turbo": 14.0,
    "Qwen/Qwen-Image-Edit-2511": 22.0,
    "unsloth/gemma-4-31B-it-unsloth-bnb-4bit": 20.0,
}


def check_disk_space(models: list[HFModel], dest: Path) -> tuple[bool, float, float]:
    needed = sum(APPROX_DOWNLOAD_GB.get(m.repo_id, 15.0) for m in models if m.required)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free / (1024**3)
    return free >= needed, needed, free


def download_model(m: HFModel, dest_root: Path, max_retries: int = 3) -> Path:
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
                # Resumable by default (huggingface_hub caches partial
                # files under blobs/ and resumes on re-run) — a killed
                # download or a flaky network is safe to just re-invoke,
                # not a reason to wipe and restart from zero.
                max_workers=4,
            )
            emit("model_download_complete", tier=m.tier, repo_id=m.repo_id, path=path, attempt=attempt)
            return Path(path)
        except HfHubHTTPError as e:
            last_exc = e
            emit(
                "model_download_retry",
                tier=m.tier, repo_id=m.repo_id, attempt=attempt, max_retries=max_retries,
                error=str(e),
            )
            time.sleep(min(2 ** attempt, 30))
        except Exception as e:  # noqa: BLE001 — surfaced to the caller either way
            last_exc = e
            emit(
                "model_download_retry",
                tier=m.tier, repo_id=m.repo_id, attempt=attempt, max_retries=max_retries,
                error=str(e),
            )
            time.sleep(min(2 ** attempt, 30))

    emit("model_download_failed", tier=m.tier, repo_id=m.repo_id, error=str(last_exc))
    raise RuntimeError(f"Failed to download {m.repo_id} after {max_retries} attempts: {last_exc}") from last_exc


# boris/vqgan_f16_16384 — same two static files scripts/training/
# download_vqgan.sh fetches via curl, reimplemented here with Python's
# stdlib urllib so the installer has no new dependency and so it can
# emit this project's own JSON-event-per-line contract like every other
# download in this script, instead of a shell script's plain stdout.
VQGAN_CONFIG_URL = "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/configs/model.yaml&dl=1"
VQGAN_CKPT_URL = "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/ckpts/last.ckpt&dl=1"


def download_vqgan(dest_dir: Path, max_retries: int = 3) -> tuple[Path, Path]:
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest_dir / "last.ckpt"
    config_path = dest_dir / "model.yaml"

    for label, url, path in [("model.yaml", VQGAN_CONFIG_URL, config_path),
                              ("last.ckpt", VQGAN_CKPT_URL, ckpt_path)]:
        if path.exists() and path.stat().st_size > 0:
            emit("vqgan_file_already_present", file=label, path=str(path))
            continue
        emit("vqgan_download_start", file=label, url=url, dest=str(path))
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                # Streamed, not read() in one shot — last.ckpt is a real
                # multi-hundred-MB checkpoint file.
                with urllib.request.urlopen(url, timeout=60) as resp, open(path, "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
                emit("vqgan_download_complete", file=label, path=str(path), attempt=attempt)
                break
            except Exception as e:  # noqa: BLE001 — retried, then surfaced to the caller
                last_exc = e
                emit("vqgan_download_retry", file=label, attempt=attempt, max_retries=max_retries, error=str(e))
                time.sleep(min(2 ** attempt, 30))
        else:
            emit("vqgan_download_failed", file=label, error=str(last_exc))
            raise RuntimeError(f"Failed to download {label} after {max_retries} attempts: {last_exc}") from last_exc

    return ckpt_path, config_path


def locate_local_artifact(
    label: str, candidate: str | None, search_globs: list[str], required: bool
) -> Path | None:
    """Local (this-repo-trained) artifacts are never downloaded — they
    either exist on disk from a real training run or they don't. This
    only verifies/discovers, matching sketch_backend.py's own
    fail-loudly-if-missing behavior rather than silently proceeding
    without a checkpoint the tier can't actually run without."""
    if candidate:
        p = Path(candidate).expanduser().resolve()
        if p.exists():
            emit("local_artifact_found", label=label, path=str(p), source="explicit")
            return p
        emit("local_artifact_missing", label=label, path=str(p), source="explicit", required=required)
        if required:
            raise FileNotFoundError(f"{label}: explicitly given path does not exist: {p}")
        return None

    # No explicit path given — best-effort discovery under common
    # training output locations, most-recently-modified wins.
    matches: list[Path] = []
    for pattern in search_globs:
        matches.extend(REPO_ROOT.glob(pattern))
    matches = [m for m in matches if m.exists()]
    if matches:
        best = max(matches, key=lambda p: p.stat().st_mtime)
        emit("local_artifact_found", label=label, path=str(best), source="auto_discovered",
             candidates_considered=len(matches))
        return best

    emit("local_artifact_missing", label=label, source="auto_discovered", required=required)
    if required:
        raise FileNotFoundError(
            f"{label}: no trained artifact found under this repo's training output "
            f"directories, and none was given explicitly. Train it first, or pass "
            f"--{label.lower().replace('_', '-')} <path>."
        )
    return None


def write_env_file(resolved: dict[str, str], low_vram: bool) -> None:
    lines = [
        "# Generated by scripts/inference/download_weights.py — do not edit by hand.",
        "# Source this before starting the service:",
        "#   set -a; source .env.inference; set +a; ./scripts/inference/run_service.sh",
        "KRISNA_USE_REAL_BACKENDS=1",
        f"KRISNA_LOW_VRAM_MODE={'1' if low_vram else '0'}",
    ]
    for k, v in resolved.items():
        if v is not None:
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("env_file_written", path=str(ENV_FILE), vars=len(resolved))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(REPO_ROOT / "weights"),
                     help="Where HF snapshots land (default: ./weights/). Independent of the "
                          "shared HF cache — kept as real local dirs so paths in .env.inference "
                          "are stable even if HF_HOME changes later.")
    ap.add_argument("--low-vram", action="store_true",
                     help="Also write KRISNA_LOW_VRAM_MODE=1 (see model_registry.py's "
                          "LOW_VRAM_REGISTRY) — does not change WHAT is downloaded, only the "
                          "runtime env this writes.")
    ap.add_argument("--fast-planner", action="store_true", help="Download Qwen3.5-4B instead of 9B.")
    ap.add_argument("--skip-quality", action="store_true",
                     help="Skip Qwen-Image-Edit-2511 (16-22GB). Polish tier falls back to "
                          "Z-Image-Turbo only — fine if you never pass quality=true to /finalize.")
    ap.add_argument("--skip-critic", action="store_true",
                     help="Skip Gemma 4 (needs its own venv-critic anyway — see "
                          "scripts/training/setup_env_critic.sh). Critique pass is unavailable "
                          "until this is downloaded and that venv exists.")
    ap.add_argument("--sketch-checkpoint", default=None,
                     help="Path to this project's trained Sketch tier checkpoint "
                          "(sketch/checkpoint_io.py's save_checkpoint(), called from "
                          "sketch/train.py). Auto-discovered under "
                          "checkpoints/sketch_stage{2,1}_*/checkpoint_final.pt if not given "
                          "(falls back to the latest checkpoint_step*.pt if no final exists).")
    ap.add_argument("--polish-lora", default=None,
                     help="Path to this project's trained Z-Image-Turbo LoRA adapter directory "
                          "(train_polish_dpo.py's <output-dir>/final/, or "
                          "train_dreambooth_lora_z_image.py's <output_dir>/ directly if DPO "
                          "hasn't run yet). Auto-discovered under models/dpo_checkpoints/**/final/ "
                          "and checkpoints/polish_default_lora*/ if not given, most-recent wins. "
                          "Optional — Polish Default runs on the frozen base without it.")
    ap.add_argument("--critic-worker-python", default="./venv-critic/bin/python",
                     help="Path to the Critic tier's isolated venv interpreter (see "
                          "scripts/training/setup_env_critic.sh). Only recorded, not created.")
    ap.add_argument("--vqgan-checkpoint", default=None,
                     help="Path to boris/vqgan_f16_16384's last.ckpt (the VQGAN decoder the "
                          "Sketch tier's tokens are decoded through at Finalize — see "
                          "orchestrator/service.py's VQTokenizer construction, which reads "
                          "KRISNA_VQGAN_CHECKPOINT/KRISNA_VQGAN_CONFIG). Auto-discovered under "
                          "checkpoints/vqgan/ (scripts/training/download_vqgan.sh's own default "
                          "destination) if not given; downloaded automatically if not found "
                          "there either, since without it every real /finalize call fails with "
                          "PIL.UnidentifiedImageError before this was even understood as a "
                          "missing-checkpoint problem (see docs/review/19_agentic_workflow_io_audit.md) "
                          "— and, until this option existed, NOTHING in the installer flow ever "
                          "surfaced that requirement at all.")
    ap.add_argument("--vqgan-config", default=None,
                     help="Path to boris/vqgan_f16_16384's model.yaml — see --vqgan-checkpoint.")
    ap.add_argument("--skip-vqgan", action="store_true",
                     help="Skip the VQGAN decoder. Finalize will fail at runtime without it — "
                          "only pass this if you already manage KRISNA_VQGAN_CHECKPOINT/"
                          "KRISNA_VQGAN_CONFIG some other way.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Resolve and report everything (disk space, HF repos, local artifacts) "
                          "without downloading or writing .env.inference.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    models = frozen_models(args.low_vram, args.fast_planner, args.skip_quality, args.skip_critic)
    required_models = [m for m in models if m.required]

    emit("plan", dest=str(dest), models=[m.repo_id for m in required_models],
         low_vram=args.low_vram, dry_run=args.dry_run)

    ok, needed_gb, free_gb = check_disk_space(models, dest)
    emit("disk_check", needed_gb=round(needed_gb, 1), free_gb=round(free_gb, 1), ok=ok)
    if not ok and not args.dry_run:
        emit("aborted", reason="insufficient_disk_space", needed_gb=round(needed_gb, 1), free_gb=round(free_gb, 1))
        return 1

    resolved_paths: dict[str, str] = {}
    failures: list[str] = []

    if not args.dry_run:
        for m in required_models:
            try:
                path = download_model(m, dest)
                resolved_paths[f"KRISNA_{m.key}_PATH"] = str(path)
            except Exception as e:  # noqa: BLE001 — collected, reported at the end
                failures.append(f"{m.repo_id}: {e}")
    else:
        for m in required_models:
            emit("model_download_skipped_dry_run", tier=m.tier, repo_id=m.repo_id)

    # Local, this-repo-trained artifacts — never downloaded, only verified.
    # Real save-path conventions, verified against the actual training
    # code/configs rather than assumed:
    #   - sketch/checkpoint_io.py's save_checkpoint(), called from
    #     sketch/train.py, writes <config.output_dir>/checkpoint_final.pt.
    #     configs/sketch_train_stage{1,2}_*.yaml set output_dir to
    #     ./checkpoints/sketch_stage{1,2}_*, relative to the repo root
    #     (scripts/training/train_sketch_stage*.sh run from there).
    #   - Two different scripts can produce the Polish-Default LoRA, and
    #     their output shapes differ:
    #       (a) train_polish_default_lora.sh (diffusers' own
    #           train_dreambooth_lora_z_image.py) saves the adapter
    #           directly IN output_dir (configs/polish_default_lora_z_image
    #           .yaml: ./checkpoints/polish_default_lora/) — no "final"
    #           subdirectory, per that script's own printed instructions.
    #       (b) train_polish_dpo.sh (train_dpo.py) saves to
    #           <output-dir>/final/ (configs/dpo_z_image_stage1_general.yaml:
    #           models/dpo_checkpoints/stage1_general/) — DPO is the
    #           refinement step on top of (a), so its output is preferred
    #           when both exist; most-recently-modified naturally wins
    #           below without needing to special-case which script ran.
    try:
        sketch_ckpt = locate_local_artifact(
            "SKETCH_CHECKPOINT", args.sketch_checkpoint,
            [
                "checkpoints/sketch_stage2_*/checkpoint_final.pt",
                "checkpoints/sketch_stage1_*/checkpoint_final.pt",
                "checkpoints/sketch_stage*/checkpoint_step*.pt",
            ],
            required=True,
        )
        if sketch_ckpt and not args.dry_run:
            resolved_paths["KRISNA_SKETCH_CHECKPOINT"] = str(sketch_ckpt)
    except FileNotFoundError as e:
        failures.append(str(e))

    # VQGAN decoder (boris/vqgan_f16_16384) — required for Finalize to
    # produce real pixels from the Sketch tier's VQ tokens (see
    # orchestrator/service.py's VQTokenizer construction). Auto-
    # discovered under checkpoints/vqgan/ first (download_vqgan.sh's own
    # default destination — reuse it if it's already there rather than
    # re-downloading); actually downloaded here if not found, since
    # nothing in the installer flow surfaced this requirement before.
    if not args.skip_vqgan:
        vqgan_ckpt = Path(args.vqgan_checkpoint).expanduser().resolve() if args.vqgan_checkpoint else None
        vqgan_cfg = Path(args.vqgan_config).expanduser().resolve() if args.vqgan_config else None
        if vqgan_ckpt and vqgan_cfg and vqgan_ckpt.exists() and vqgan_cfg.exists():
            emit("local_artifact_found", label="VQGAN", path=str(vqgan_ckpt), source="explicit")
        else:
            default_dir = REPO_ROOT / "checkpoints" / "vqgan"
            existing_ckpt, existing_cfg = default_dir / "last.ckpt", default_dir / "model.yaml"
            if existing_ckpt.exists() and existing_cfg.exists():
                emit("local_artifact_found", label="VQGAN", path=str(existing_ckpt), source="auto_discovered")
                vqgan_ckpt, vqgan_cfg = existing_ckpt, existing_cfg
            elif not args.dry_run:
                try:
                    vqgan_ckpt, vqgan_cfg = download_vqgan(default_dir)
                except Exception as e:  # noqa: BLE001 — collected, reported at the end
                    failures.append(f"VQGAN checkpoint: {e}")
                    vqgan_ckpt = vqgan_cfg = None
            else:
                emit("model_download_skipped_dry_run", tier="vqgan", repo_id="boris/vqgan_f16_16384")
                vqgan_ckpt = vqgan_cfg = None
        if vqgan_ckpt and vqgan_cfg and not args.dry_run:
            resolved_paths["KRISNA_VQGAN_CHECKPOINT"] = str(vqgan_ckpt)
            resolved_paths["KRISNA_VQGAN_CONFIG"] = str(vqgan_cfg)

    # LoRA is genuinely optional — Polish Default runs frozen without it,
    # just without the project's own fine-tune (see factory.py's comment
    # on lora_adapter_path being the one adapter this project keeps).
    polish_lora = locate_local_artifact(
        "POLISH_LORA", args.polish_lora,
        [
            "models/dpo_checkpoints/**/final/adapter_model.safetensors",
            "checkpoints/polish_default_lora*/adapter_model.safetensors",
        ],
        required=False,
    )
    if polish_lora and not args.dry_run:
        resolved_paths["KRISNA_POLISH_DEFAULT_LORA_PATH"] = str(polish_lora.parent)

    if not args.skip_critic and not args.dry_run:
        worker_python = Path(args.critic_worker_python)
        if worker_python.exists():
            emit("critic_venv_found", path=str(worker_python))
            resolved_paths["KRISNA_CRITIC_VENV_PYTHON"] = str(worker_python)
        else:
            emit("critic_venv_missing", path=str(worker_python),
                 hint="run ./scripts/training/setup_env_critic.sh")
            failures.append(
                f"Critic worker venv not found at {worker_python} — run "
                "./scripts/training/setup_env_critic.sh, or pass --skip-critic."
            )

    if args.dry_run:
        emit("dry_run_complete", would_fail=failures)
        return 1 if failures else 0

    if failures:
        emit("install_incomplete", failures=failures)
        # Still write whatever succeeded — a partial .env.inference with
        # comments-visible gaps is more useful than nothing, and re-running
        # this script only re-does what's missing (HF downloads resume;
        # local-artifact checks are idempotent).
        write_env_file(resolved_paths, args.low_vram)
        return 1

    write_env_file(resolved_paths, args.low_vram)
    emit("install_complete", env_file=str(ENV_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
