"""Unit tests for Krisna Inference hardware and deployment preflight diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from krisna_inference.common.hardware import (
    ConfigStatus,
    HardwareIncompatibleError,
    HardwareStatus,
    check_gpu_cuda_match,
    check_inference_config,
    find_repo_root,
    format_diagnostic_report,
    main,
)


def test_find_repo_root():
    root = find_repo_root()
    assert (root / "pytest.ini").exists() or (root / "inference").is_dir()


def test_cpu_only_returns_incompatible():
    with patch("torch.cuda.is_available", return_value=False):
        status = check_gpu_cuda_match(require_gpu=False)
        assert status.has_gpu is False
        assert status.cuda_available is False
        assert status.is_compatible is False
        assert len(status.errors) > 0
        assert any("CUDA" in err or "GPU" in err for err in status.errors)


def test_cpu_only_raises_when_gpu_required():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(HardwareIncompatibleError) as excinfo:
            check_gpu_cuda_match(require_gpu=True)
        assert "ERRORS (DEPLOYMENT BLOCKED)" in str(excinfo.value)
        assert "ACTIONABLE REMEDIATION" in str(excinfo.value)


def test_mock_gpu_compatible():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX A6000"), \
         patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
         patch("torch.cuda.mem_get_info", return_value=(45 * (1024**3), 48 * (1024**3))):

        status = check_gpu_cuda_match(require_gpu=True, min_vram_gb=16.0, low_vram=False)
        assert status.has_gpu is True
        assert status.cuda_available is True
        assert status.is_compatible is True
        assert status.device_name == "NVIDIA RTX A6000"
        assert status.compute_capability == (8, 6)
        assert status.total_vram_gb >= 47.9
        assert status.free_vram_gb >= 44.9
        assert len(status.errors) == 0


def test_low_vram_mode_warnings_and_thresholds():
    # 12GB VRAM card in standard mode emits warning recommending low-VRAM
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4070"), \
         patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
         patch("torch.cuda.mem_get_info", return_value=(11 * (1024**3), 12 * (1024**3))):

        status_std = check_gpu_cuda_match(require_gpu=False, low_vram=False)
        assert any("Consider enabling low-VRAM mode" in w for w in status_std.warnings)
        assert status_std.is_compatible is True  # Warning, not blocking error in std mode

        # In low-VRAM mode, 12GB meets the 11.5GB requirement
        status_low = check_gpu_cuda_match(require_gpu=False, low_vram=True)
        assert not any("Consider enabling low-VRAM mode" in w for w in status_low.warnings)
        assert status_low.is_compatible is True


def test_insufficient_vram_in_low_vram_mode_blocks():
    # 8GB VRAM card fails low-VRAM target (11.5GB)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3070"), \
         patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
         patch("torch.cuda.mem_get_info", return_value=(7 * (1024**3), 8 * (1024**3))):

        status = check_gpu_cuda_match(require_gpu=False, low_vram=True)
        assert status.is_compatible is False
        assert any("below the recommended envelope" in e for e in status.errors)


def test_compute_capability_warning_older_arch():
    # sm_70 (Volta) is below sm_75
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla V100"), \
         patch("torch.cuda.get_device_capability", return_value=(7, 0)), \
         patch("torch.cuda.mem_get_info", return_value=(30 * (1024**3), 32 * (1024**3))):

        status = check_gpu_cuda_match(require_gpu=False)
        assert any("below 7.5" in w for w in status.warnings)


def test_check_inference_config_discovery(tmp_path):
    # Test fake repo root
    (tmp_path / ".env.inference").write_text("KRISNA_PORT=8420\n", encoding="utf-8")
    critic_bin = tmp_path / "venv-critic" / "bin" / "python"
    critic_bin.parent.mkdir(parents=True)
    critic_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    critic_bin.chmod(0o755)

    cfg = check_inference_config(require_real=False, repo_root=tmp_path)
    assert cfg.env_inference_exists is True
    assert cfg.critic_python_exists is True


def test_diagnostic_report_formatting():
    hw = HardwareStatus(
        has_gpu=False,
        is_compatible=False,
        errors=["No NVIDIA GPU detected"],
        warnings=["Low host RAM"],
    )
    cfg = ConfigStatus(use_real_backends=True)
    report = format_diagnostic_report(hw, cfg)
    assert "Krisna Inference" in report
    assert "GPU Detected         : NO" in report
    assert "ERRORS (DEPLOYMENT BLOCKED)" in report
    assert "ACTIONABLE REMEDIATION" in report
    assert "NVIDIA Container Toolkit" in report


def test_cli_diagnostic_json():
    root = find_repo_root()
    script = root / "scripts" / "inference" / "check_hardware.py"
    res = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    assert "hardware" in data
    assert "config" in data
    assert "has_gpu" in data["hardware"]
    assert "is_compatible" in data["hardware"]
