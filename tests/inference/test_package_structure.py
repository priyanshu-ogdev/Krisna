"""Structural and integrity tests for the krisna-inference package.

Verifies:
1. Package version and public re-exports in krisna_inference/__init__.py.
2. CLI argument parsing in krisna_inference.cli.
3. Service CLI entrypoint function in krisna_inference.orchestrator.service.
4. Backward-compatible runner in inference-runtime/run_agentic_session.py.
5. Cleanliness of inference directory (no stray duplicate databases).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_package_exports():
    """Verify krisna_inference package exports its canonical symbols."""
    import krisna_inference

    assert hasattr(krisna_inference, "__version__")
    assert krisna_inference.__version__ == "0.2.0"
    assert krisna_inference.__version_info__ == (0, 2, 0)

    # Core classes
    from krisna_inference import (
        DesignState,
        DesignStateStore,
        MockBackend,
        ModelBackend,
        ModelSpec,
        ResidencyState,
        SessionStage,
        StateMachine,
        SwapOrchestrator,
        Tier,
        real_backend_factory,
    )

    assert SwapOrchestrator is not None
    assert DesignState is not None
    assert SessionStage is not None
    assert DesignStateStore is not None
    assert Tier.PLANNER == "planner"
    assert ResidencyState.IDLE_RESIDENT == "idle_resident"
    assert MockBackend is not None
    assert callable(real_backend_factory)


def test_cli_parser():
    """Verify krisna_inference.cli argument parser configuration."""
    from krisna_inference.cli import build_parser

    parser = build_parser()
    # Test safe default parsing
    args = parser.parse_args([])
    assert args.real is False
    assert args.low_vram is False
    assert args.finalize is False
    assert args.finalize_tier == "polish_default"

    # Test flags
    args = parser.parse_args(["--real", "--low-vram", "--finalize", "--finalize-tier", "polish_quality"])
    assert args.real is True
    assert args.low_vram is True
    assert args.finalize is True
    assert args.finalize_tier == "polish_quality"


def test_service_main_entrypoint():
    """Verify service.py exposes a main() function for CLI execution."""
    from krisna_inference.orchestrator import service

    assert hasattr(service, "main")
    assert callable(service.main)


def test_no_stray_databases_in_inference():
    """Verify no stray SQLite database files are left inside the inference package."""
    stray_root = _REPO_ROOT / "inference" / "krisna_preference_pairs.db"
    stray_src = _REPO_ROOT / "inference" / "src" / "krisna_preference_pairs.db"

    assert not stray_root.exists(), f"Stray database found: {stray_root}"
    assert not stray_src.exists(), f"Stray database found: {stray_src}"


def test_runtime_cli_harness():
    """Verify inference/runtime/run_agentic_session.py is intact and executable."""
    forwarder = _REPO_ROOT / "inference" / "runtime" / "run_agentic_session.py"
    assert forwarder.exists()

    # Run --help to verify clean invocation
    result = subprocess.run(
        [sys.executable, str(forwarder), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "krisna-session" in result.stdout or "Inference Agentic Session" in result.stdout


def test_unified_inference_frontend_structure():
    """Verify inference/frontend contains package.json, server.js, and public assets."""
    frontend_dir = _REPO_ROOT / "inference" / "frontend"
    assert (frontend_dir / "package.json").exists()
    assert (frontend_dir / "server.js").exists()
    assert (frontend_dir / "public" / "index.html").exists()
    assert (frontend_dir / "public" / "app.js").exists()
    assert (frontend_dir / "public" / "style.css").exists()


def test_no_obsolete_sprawl_directories():
    """Verify legacy fragmented directories do not exist at repository root."""
    assert not (_REPO_ROOT / "src").exists(), "Obsolete src/ placeholder directory must not exist"
    assert not (_REPO_ROOT / "inference-frontend").exists(), "Legacy inference-frontend/ must be under inference/frontend"
    assert not (_REPO_ROOT / "inference-runtime").exists(), "Legacy inference-runtime/ must be under inference/runtime"

