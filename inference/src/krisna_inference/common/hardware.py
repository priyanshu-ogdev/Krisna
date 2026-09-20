"""Hardware and deployment configuration verification for Krisna Inference.

Validates:
1. NVIDIA GPU presence and driver / CUDA compatibility.
2. PyTorch CUDA capability, compute capability (sm_75+ required, sm_80+ recommended),
   and VRAM envelope sufficiency (16GB+ default, 11.5GB+ low-VRAM).
3. Configuration readiness: weights, checkpoints, and isolated Critic venv python.
4. Fail-fast diagnostics: If real backends are requested and hardware is inadequate,
   fails loudly with actionable remediation steps for Linux, Windows, and Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class HardwareIncompatibleError(RuntimeError):
    """Raised when hardware does not meet requirements for real backend deployment."""
    pass


@dataclass
class HardwareStatus:
    has_gpu: bool = False
    gpu_count: int = 0
    device_name: str | None = None
    cuda_available: bool = False
    cuda_driver_version: str | None = None
    cuda_runtime_version: str | None = None
    cudnn_version: str | None = None
    compute_capability: tuple[int, int] | None = None
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    is_compatible: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_gpu": self.has_gpu,
            "gpu_count": self.gpu_count,
            "device_name": self.device_name,
            "cuda_available": self.cuda_available,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cudnn_version": self.cudnn_version,
            "compute_capability": (
                f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability
                else None
            ),
            "total_vram_gb": round(self.total_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "is_compatible": self.is_compatible,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ConfigStatus:
    is_ready: bool = False
    use_real_backends: bool = False
    low_vram_mode: bool = False
    critic_python_path: str | None = None
    critic_python_exists: bool = False
    sketch_checkpoint: str | None = None
    sketch_checkpoint_exists: bool = False
    polish_lora_path: str | None = None
    polish_lora_exists: bool = False
    env_inference_exists: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "use_real_backends": self.use_real_backends,
            "low_vram_mode": self.low_vram_mode,
            "critic_python_path": self.critic_python_path,
            "critic_python_exists": self.critic_python_exists,
            "sketch_checkpoint": self.sketch_checkpoint,
            "sketch_checkpoint_exists": self.sketch_checkpoint_exists,
            "polish_lora_path": self.polish_lora_path,
            "polish_lora_exists": self.polish_lora_exists,
            "env_inference_exists": self.env_inference_exists,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _probe_system_ram() -> tuple[float, float]:
    """Returns (total_gb, available_gb) for host system RAM."""
    # 1. Try /proc/meminfo on Linux
    if Path("/proc/meminfo").exists():
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    k, _, v = line.partition(":")
                    info[k.strip()] = v.strip()
            total_kb = int(info["MemTotal"].split()[0])
            avail_kb = int(info.get("MemAvailable", info.get("MemFree", "0")).split()[0])
            return total_kb / (1024**2), avail_kb / (1024**2)
        except Exception:
            pass

    # 2. Try psutil if available
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except Exception:
        pass

    # 3. Try Windows ctypes GlobalMemoryStatusEx
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024**3), stat.ullAvailPhys / (1024**3)
        except Exception:
            pass

    return 0.0, 0.0


def check_gpu_cuda_match(
    require_gpu: bool = False,
    min_vram_gb: float = 11.5,
    low_vram: bool = False,
) -> HardwareStatus:
    """Probes GPU, CUDA runtime, driver compatibility, and VRAM capacity.

    Args:
        require_gpu: If True and no GPU is found or CUDA fails, raises HardwareIncompatibleError.
        min_vram_gb: Minimum required physical VRAM in gigabytes.
        low_vram: Whether deployment is targeting low-VRAM mode (11.5-12GB).
    """
    status = HardwareStatus()
    status.total_ram_gb, status.available_ram_gb = _probe_system_ram()

    # Check for nvidia-smi tool availability
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    if has_nvidia_smi:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                status.cuda_driver_version = res.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

    # Probe PyTorch CUDA integration
    try:
        import torch

        status.cuda_available = bool(torch.cuda.is_available())
        if status.cuda_available:
            status.has_gpu = True
            status.gpu_count = torch.cuda.device_count()
            status.device_name = torch.cuda.get_device_name(0)
            status.cuda_runtime_version = getattr(torch.version, "cuda", None)
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                status.cudnn_version = str(torch.backends.cudnn.version())

            # Check compute capability
            cc = torch.cuda.get_device_capability(0)
            status.compute_capability = cc

            # VRAM measurement
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            status.free_vram_gb = free_bytes / (1024**3)
            status.total_vram_gb = total_bytes / (1024**3)

            # Compute capability checks: sm_75 is Turing (minimum for bnb NF4 / bfloat16 emulation),
            # sm_80+ is Ampere/Ada/Hopper/Blackwell (native bfloat16 and fast tensor cores).
            major, minor = cc
            if major < 7 or (major == 7 and minor < 5):
                status.warnings.append(
                    f"GPU compute capability ({major}.{minor}) is below 7.5 (Turing). "
                    "4-bit NF4 bitsandbytes quantization and bfloat16 operations may degrade or fail."
                )
            elif major < 8:
                status.warnings.append(
                    f"GPU compute capability ({major}.{minor}) is Turing (sm_75). "
                    "Ampere or newer (sm_80+) is recommended for optimal native BF16 performance."
                )

            # VRAM sufficiency checks
            effective_target = min_vram_gb if low_vram else max(16.0, min_vram_gb)
            if status.total_vram_gb < effective_target:
                msg = (
                    f"GPU VRAM ({status.total_vram_gb:.1f} GB) is below the recommended "
                    f"envelope ({effective_target:.1f} GB for {'low-VRAM' if low_vram else 'standard'} mode)."
                )
                if low_vram:
                    status.errors.append(msg)
                else:
                    status.warnings.append(msg + " Consider enabling low-VRAM mode (KRISNA_LOW_VRAM_MODE=1).")

            # System RAM check for low-VRAM offloading
            if low_vram and status.total_ram_gb > 0 and status.total_ram_gb < 32.0:
                status.warnings.append(
                    f"System RAM ({status.total_ram_gb:.1f} GB) is modest for low-VRAM mode. "
                    "CPU offload for Polish and Critic models functions best with >= 32GB (ideally 64-128GB)."
                )
        else:
            if has_nvidia_smi:
                status.errors.append(
                    "NVIDIA GPU driver detected via nvidia-smi, but PyTorch cannot access CUDA. "
                    "PyTorch was likely installed without CUDA support (CPU-only build)."
                )
            else:
                status.errors.append(
                    "No NVIDIA GPU or CUDA driver detected on this system."
                )
    except ImportError:
        status.errors.append("PyTorch is not installed in the current environment.")

    # Determine overall compatibility
    if status.has_gpu and status.cuda_available and not status.errors:
        status.is_compatible = True
    else:
        status.is_compatible = False

    if require_gpu and not status.is_compatible:
        report = format_diagnostic_report(status)
        raise HardwareIncompatibleError(report)

    return status


def find_repo_root(start: Path | None = None) -> Path:
    """Discovers the monorepo root by looking for pytest.ini or scripts+inference directories."""
    cur = (start or Path(__file__)).resolve()
    for p in [cur] + list(cur.parents):
        if (p / "pytest.ini").exists() or ((p / "scripts").is_dir() and (p / "inference").is_dir()):
            return p
    return Path.cwd()


def check_inference_config(
    require_real: bool = False,
    repo_root: Path | None = None,
) -> ConfigStatus:
    """Validates configuration files, model weights, and isolated Critic python."""
    root = repo_root or find_repo_root()
    cfg = ConfigStatus()

    cfg.use_real_backends = os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1"
    cfg.low_vram_mode = os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"

    # Check .env.inference
    env_file = root / ".env.inference"
    cfg.env_inference_exists = env_file.is_file()

    # Check Critic worker Python
    critic_py_env = os.environ.get("KRISNA_CRITIC_VENV_PYTHON")
    if critic_py_env:
        critic_path = Path(critic_py_env)
    else:
        candidates = [
            root / "venv-critic" / "Scripts" / "python.exe",
            root / "venv-critic" / "bin" / "python",
        ]
        found = next((c for c in candidates if c.is_file()), None)
        if found:
            critic_path = found
        else:
            critic_path = (
                root / "venv-critic" / "Scripts" / "python.exe"
                if sys.platform == "win32"
                else root / "venv-critic" / "bin" / "python"
            )
    cfg.critic_python_path = str(critic_path)
    cfg.critic_python_exists = critic_path.is_file()

    # Check Sketch checkpoint if configured
    sketch_ckpt = os.environ.get("KRISNA_SKETCH_CHECKPOINT")
    if sketch_ckpt:
        cfg.sketch_checkpoint = sketch_ckpt
        cfg.sketch_checkpoint_exists = Path(sketch_ckpt).exists()

    # Check Polish LoRA if configured
    polish_lora = os.environ.get("KRISNA_POLISH_DEFAULT_LORA_PATH")
    if polish_lora:
        cfg.polish_lora_path = polish_lora
        cfg.polish_lora_exists = Path(polish_lora).exists()

    # Real deployment validation
    if require_real or cfg.use_real_backends:
        if not cfg.critic_python_exists:
            cfg.warnings.append(
                f"Critic isolated Python interpreter not found at: {cfg.critic_python_path}. "
                "The on-demand Critic tier (Gemma-4) will not be available until created. "
                "Run ./setup.sh critic or ensure KRISNA_CRITIC_VENV_PYTHON is set."
            )
        if sketch_ckpt and not cfg.sketch_checkpoint_exists:
            cfg.errors.append(f"Configured KRISNA_SKETCH_CHECKPOINT does not exist: {sketch_ckpt}")
        if polish_lora and not cfg.polish_lora_exists:
            cfg.errors.append(f"Configured KRISNA_POLISH_DEFAULT_LORA_PATH does not exist: {polish_lora}")

    cfg.is_ready = len(cfg.errors) == 0
    return cfg


def format_diagnostic_report(hw: HardwareStatus, cfg: ConfigStatus | None = None) -> str:
    """Builds a human-readable, actionable diagnostic report."""
    lines = [
        "=================================================================",
        "        Krisna Inference -- Deployment & Hardware Diagnostics    ",
        "=================================================================",
    ]

    # Hardware Section
    lines.append(f"GPU Detected         : {'YES (' + (hw.device_name or 'Unknown') + ')' if hw.has_gpu else 'NO'}")
    if hw.has_gpu:
        lines.append(f"GPU Count            : {hw.gpu_count}")
        lines.append(f"Driver Version       : {hw.cuda_driver_version or 'N/A'}")
        lines.append(f"CUDA Runtime         : {hw.cuda_runtime_version or 'N/A'}")
        lines.append(f"cuDNN Version        : {hw.cudnn_version or 'N/A'}")
        lines.append(f"Compute Capability   : {hw.compute_capability[0]}.{hw.compute_capability[1]}" if hw.compute_capability else "Compute Capability   : N/A")
        lines.append(f"Total VRAM           : {hw.total_vram_gb:.2f} GB")
        lines.append(f"Free VRAM            : {hw.free_vram_gb:.2f} GB")
    lines.append(f"System RAM           : {hw.total_ram_gb:.2f} GB (Available: {hw.available_ram_gb:.2f} GB)")
    lines.append(f"Hardware Compatible  : {'YES' if hw.is_compatible else 'NO'}")

    # Config Section
    if cfg:
        lines.append("-----------------------------------------------------------------")
        lines.append(f"Real Backends Mode   : {'ENABLED' if cfg.use_real_backends else 'MOCK (Test Only)'}")
        lines.append(f"Low-VRAM Offload     : {'ENABLED' if cfg.low_vram_mode else 'DISABLED'}")
        lines.append(f"Critic Isolated venv : {'FOUND' if cfg.critic_python_exists else 'MISSING'} ({cfg.critic_python_path})")
        if cfg.sketch_checkpoint:
            lines.append(f"Sketch Checkpoint    : {'FOUND' if cfg.sketch_checkpoint_exists else 'MISSING'} ({cfg.sketch_checkpoint})")
        if cfg.polish_lora_path:
            lines.append(f"Polish LoRA Adapter  : {'FOUND' if cfg.polish_lora_exists else 'MISSING'} ({cfg.polish_lora_path})")

    # Errors & Remediation
    all_errors = list(hw.errors) + (cfg.errors if cfg else [])
    all_warnings = list(hw.warnings) + (cfg.warnings if cfg else [])

    if all_warnings:
        lines.append("-----------------------------------------------------------------")
        lines.append("WARNINGS:")
        for w in all_warnings:
            lines.append(f"  [!] {w}")

    if all_errors:
        lines.append("-----------------------------------------------------------------")
        lines.append("ERRORS (DEPLOYMENT BLOCKED):")
        for e in all_errors:
            lines.append(f"  [X] {e}")
        lines.append("")
        lines.append("ACTIONABLE REMEDIATION:")
        if not hw.has_gpu or not hw.cuda_available:
            lines.append("  1. Bare Metal (Linux/Windows):")
            lines.append("     - Install NVIDIA Driver (>= 535 recommended).")
            lines.append("     - Install CUDA PyTorch: pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124")
            lines.append("  2. Docker Deployment:")
            lines.append("     - Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/")
            lines.append("     - Ensure container is launched with GPU capability:")
            lines.append("       docker run --gpus all -p 8420:8420 krisna-inference:latest")
            lines.append("       or use docker compose up (GPU reservations pre-configured).")
            lines.append("  3. Automated Testing / Mock Development (No GPU):")
            lines.append("     - Explicitly run in mock mode: export KRISNA_USE_REAL_BACKENDS=0")

    lines.append("=================================================================")
    return "\n".join(lines)


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Krisna Inference Hardware & Configuration Diagnostic Tool",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Fail with exit code 1 if GPU/CUDA is missing or incompatible")
    parser.add_argument("--deploy-check", action="store_true", help="Enforce real-backend deployment readiness (GPU + config)")
    parser.add_argument("--low-vram", action="store_true", help="Evaluate against low-VRAM threshold (11.5GB)")
    parser.add_argument("--json", action="store_true", help="Output diagnostic results as JSON")
    args = parser.parse_args()

    require_gpu = args.require_gpu or args.deploy_check
    low_vram = args.low_vram or os.environ.get("KRISNA_LOW_VRAM_MODE") == "1"

    hw = check_gpu_cuda_match(require_gpu=False, low_vram=low_vram)
    cfg = check_inference_config(require_real=args.deploy_check)

    if args.json:
        out = {"hardware": hw.to_dict(), "config": cfg.to_dict()}
        print(json.dumps(out, indent=2))
    else:
        print(format_diagnostic_report(hw, cfg))

    if require_gpu and not hw.is_compatible:
        return 1
    if args.deploy_check and (not hw.is_compatible or not cfg.is_ready):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
