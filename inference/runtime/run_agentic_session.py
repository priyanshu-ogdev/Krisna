#!/usr/bin/env python3
"""Inference-runtime: CLI harness for running a real agentic session end-to-end.

This script is maintained for backward compatibility. It delegates directly
to the canonical package CLI entry point: `krisna_inference.cli.main()`.

Usage:
    python inference-runtime/run_agentic_session.py
    python inference-runtime/run_agentic_session.py --real
    python inference-runtime/run_agentic_session.py --real --low-vram
    python inference-runtime/run_agentic_session.py --message "a minimalist login screen" --finalize
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure monorepo packages (krisna_inference, krisna_training) are importable
# when running this CLI script directly from repository root without editable install.
_CUR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR.parent.parent if (_CUR.parent.parent / "pytest.ini").exists() else _CUR.parent
for _pkg_dir in (_REPO_ROOT / "inference" / "src", _REPO_ROOT / "training" / "src"):
    if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

from krisna_inference.cli import main

if __name__ == "__main__":
    sys.exit(main())
