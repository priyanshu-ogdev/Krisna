#!/usr/bin/env python3
"""Krisna Inference — Production-Grade Local Runner.

Performs hardware preflight checks (VRAM, Host RAM, CUDA compute capabilities),
discovers local model artifacts (GGUF, LoRA), configures low-VRAM modes,
and launches the FastAPI service with health/readiness endpoints.

Usage:
    python scripts/inference/run_local.py [--host 127.0.0.1] [--port 8420] [--real] [--warm]

References & Citations:
- Gerganov et al. (2023). llama.cpp (zero-copy mmap model execution).
- Wallace et al. (CVPR 2024). Diffusion-DPO (diffusion alignment).
- PRD §3 & §7: 12GB Low-VRAM / 16GB Host RAM Consumer GPU Envelope.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project packages to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference" / "src"))
sys.path.insert(0, str(REPO_ROOT / "training" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from krisna_inference.backends.model_loader import (
    discover_mmproj_model,
    discover_planner_model,
    probe_cuda_hardware,
    verify_manifest,
)
from krisna_inference.common.hardware import _probe_system_ram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("krisna.local_runner")


def preflight_hardware() -> dict:
    """Check GPU and Host RAM feasibility against the PRD resource envelope."""
    log.info("==> Running hardware preflight checks...")
    hw = probe_cuda_hardware()
    host_ram = _probe_system_ram()

    log.info(
        "Host RAM: total=%.1f GB, available=%.1f GB",
        host_ram.total_gb,
        host_ram.available_gb,
    )

    if hw["cuda_available"]:
        log.info(
            "CUDA GPU detected: '%s' (Total VRAM: %.1f GB, Free: %.1f GB, Compute: %s)",
            hw["device_name"],
            hw["vram_total_gb"],
            hw["vram_free_gb"],
            hw.get("compute_capability", "N/A"),
        )
        if hw["supports_flash_attn"]:
            log.info("FlashAttention-2 kernel supported on this device.")
    else:
        log.warning(
            "No CUDA GPU detected or PyTorch CUDA uninitialized. "
            "Running in CPU / synthetic mock fallback mode."
        )

    return {"cuda": hw, "host_ram": host_ram}


def preflight_models() -> dict:
    """Discover local Planner and Polish models."""
    log.info("==> Discovering local model artifacts...")
    planner_gguf = discover_planner_model()
    mmproj_gguf = discover_mmproj_model()

    if planner_gguf:
        log.info("Discovered Planner GGUF: %s (%.2f GB)", planner_gguf, planner_gguf.stat().st_size / (1024 ** 3))
        manifest = planner_gguf.parent / "manifest.json"
        if manifest.exists():
            v = verify_manifest(manifest)
            if v["valid"]:
                log.info("Verified artifact manifest: %s (repo: %s)", manifest, v.get("repo_id"))
            else:
                log.warning("Manifest validation warning: %s", v.get("error"))
    else:
        log.info("No local Planner GGUF found. (Models directory: models/planner)")
        log.info(
            "To download recommended weights: "
            "python scripts/inference/download_planner_models.py --repo-id ggml-org/Qwen3.5-9B-GGUF --filename Qwen3.5-9B-Q4_K_M.gguf"
        )

    if mmproj_gguf:
        log.info("Discovered multimodal projector: %s", mmproj_gguf)

    polish_id = os.environ.get("KRISNA_POLISH_MODEL_ID", "Qwen/Qwen-Image-2.1")
    lora_path = os.environ.get("KRISNA_POLISH_LORA_PATH") or os.environ.get("KRISNA_POLISH_DEFAULT_LORA_PATH")
    log.info("Polish model ID: %s", polish_id)
    if lora_path and Path(lora_path).exists():
        log.info("Polish LoRA adapter: %s", lora_path)

    return {
        "planner_gguf": planner_gguf,
        "mmproj_gguf": mmproj_gguf,
        "polish_id": polish_id,
        "lora_path": lora_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Krisna Inference Local Runner")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    parser.add_argument("--real", action="store_true", help="Enable real backends (requires GPU and weights)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--warm", action="store_true", help="Pre-warm baseline tiers at startup")
    args = parser.parse_args()

    hw_info = preflight_hardware()
    models_info = preflight_models()

    # Automatically calibrate low-VRAM mode
    cuda_hw = hw_info["cuda"]
    if cuda_hw["cuda_available"] and cuda_hw["vram_total_gb"] < 16.0:
        log.info("VRAM < 16GB (%.1f GB). Auto-enabling KRISNA_LOW_VRAM_MODE=1.", cuda_hw["vram_total_gb"])
        os.environ["KRISNA_LOW_VRAM_MODE"] = "1"

    if args.real:
        os.environ["KRISNA_USE_REAL_BACKENDS"] = "1"
        log.info("KRISNA_USE_REAL_BACKENDS=1 enabled.")

    if models_info["planner_gguf"]:
        os.environ["KRISNA_PLANNER_GGUF"] = str(models_info["planner_gguf"])
    if models_info["mmproj_gguf"]:
        os.environ["KRISNA_PLANNER_MMPROJ"] = str(models_info["mmproj_gguf"])

    log.info("==> Starting Krisna Inference Service on http://%s:%d ...", args.host, args.port)
    log.info("    Health endpoint:     http://%s:%d/health", args.host, args.port)
    log.info("    Readiness endpoint:  http://%s:%d/ready", args.host, args.port)
    log.info("    Model status:        http://%s:%d/models/status", args.host, args.port)
    log.info("    API documentation:   http://%s:%d/docs", args.host, args.port)

    import uvicorn

    uvicorn.run(
        "krisna_inference.orchestrator.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
