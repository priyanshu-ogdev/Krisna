#!/usr/bin/env python3
"""Krisna Inference -- Hardware and Deployment Preflight Check.

Runs preflight hardware and environment checks for Krisna Inference.
Verifies NVIDIA GPU availability, CUDA driver version, PyTorch CUDA capability,
VRAM envelope, model configuration, and isolated Critic tier virtualenv.

Usage:
    python scripts/inference/check_hardware.py
    python scripts/inference/check_hardware.py --require-gpu
    python scripts/inference/check_hardware.py --deploy-check
    python scripts/inference/check_hardware.py --low-vram
    python scripts/inference/check_hardware.py --json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add inference/src to sys.path if not already installed
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SRC = REPO_ROOT / "inference" / "src"
if INFERENCE_SRC.exists() and str(INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(INFERENCE_SRC))

try:
    from krisna_inference.common.hardware import main
except ImportError as err:
    sys.stderr.write(
        f"[ERROR] Failed to import krisna_inference.common.hardware: {err}\n"
        f"Ensure Python is running from repo root or inference package is accessible.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
