# Phase 28: Production Docker Containerization, Multi-Tier Venv Isolation, and Hardware/Config Preflight Upgrades

**Context**: Upgrades Krisna Inference into an enterprise-grade, GPU-accelerated Docker container architecture featuring strict isolation for tiers with conflicting dependencies, and overhauls the installer toolchain with robust, fail-fast GPU, CUDA, and configuration checks.

---

## 1. Problem Statement & Root Cause Analysis

### A. The Upstream Dependency Conflict in Docker
In single-process or single-virtualenv environments, upstream dependency pins between inference tiers conflict:
1. **Main Inference Tier (Planner Qwen3.5, Sketch MaskGIT, Polish Z-Image-Turbo / Qwen-Image-Edit)**:
   Requires modern `torch >= 2.6.0`, `transformers >= 5.2.0` (native Qwen3.5 support), and `diffusers >= 0.31.0`.
2. **Critic Tier (Gemma 4 31B Dense)**:
   Requires `transformers == 5.5.0` (Unsloth's `FastVisionModel` pin) along with `unsloth` and `unsloth_zoo`.

Installing both into a single Python virtualenv causes silent regressions or `ImportError` exceptions because `transformers==5.5.0` breaks or caps modern pipeline APIs while newer transformers breaks Unsloth Gemma-4 kernels.

### B. Late Failures on Incompatible Hardware
Previously, launching the inference service or running installer scripts on machines lacking NVIDIA GPUs or compatible CUDA runtimes led to cryptic PyTorch CPU errors, CUDA memory allocation hangs, or silent fallbacks during runtime inference passes. In production deploy and release postures, MockBackend is strictly reserved for automated testing / CI; real deployments must fail fast with actionable remediation guidance.

---

## 2. Architectural Solution

### A. Multi-Environment Virtualenv Isolation in Docker
We containerized the stack under `nvidia/cuda:12.4.1-runtime-ubuntu22.04` using Python 3.11 and created two distinct, isolated virtual environments inside the container filesystem:
- `/opt/venv-inference`: Hosts `krisna_inference`, FastAPI SwapOrchestrator, Planner, Sketch, and Polish models.
- `/opt/venv-critic`: Hosts Gemma 4 31B Dense and Unsloth under `transformers==5.5.0`.

**Inter-Process Communication (IPC)**:
`CriticBackend` executes `critic_worker.py` out-of-process via standard `subprocess.Popen([KRISNA_CRITIC_VENV_PYTHON, "critic_worker.py"])` over JSON stdin/stdout pipes. Setting `ENV KRISNA_CRITIC_VENV_PYTHON=/opt/venv-critic/bin/python` routes all Critic calls to the isolated environment with zero code modification.

### B. Unified Hardware & Configuration Diagnostics Engine
Implemented in `inference/src/krisna_inference/common/hardware.py` and exposed via standalone CLI `scripts/inference/check_hardware.py`:
- Probes `torch.cuda.is_available()`, `nvidia-smi` driver version, PyTorch CUDA build, device name, compute capability (sm_75+ required, sm_80+ recommended), and physical VRAM vs. declared envelope (16GB+ default, 11.5GB+ low-VRAM).
- Inspects configuration readiness (`.env.inference`, weights, checkpoints, isolated Critic venv python).
- Formats actionable remediation reports for Bare Metal (Linux/Windows) and Docker (NVIDIA Container Toolkit).

### C. Fail-Fast Enforcement at Service Startup
In `inference/src/krisna_inference/orchestrator/service.py`:
When `KRISNA_USE_REAL_BACKENDS=1`, the service validates hardware compatibility and configuration readiness before initializing the `SwapOrchestrator`. If a GPU is absent, CUDA mismatches, or critical configurations are missing, it raises `HardwareIncompatibleError` immediately and outputs the diagnostic remediation report.

### D. Docker Compose Multi-Service Architecture
Defined in root `docker-compose.yml`:
1. **`krisna-inference`**:
   - Built from `docker/Dockerfile.inference`.
   - Exposes port `8420` with `/healthz` healthchecks.
   - GPU reservations configured (`driver: nvidia, count: all, capabilities: [gpu]`).
   - Host volume mounts for `./models`, `./checkpoints`, `./weights`, and `./krisna_blobs`.
2. **`krisna-frontend`**:
   - Built from `docker/Dockerfile.frontend` (Node.js 20 Alpine).
   - Exposes Web Studio UI on port `3000`.
   - Inter-service networking connected to `http://inference:8420`.
3. **Launchers**:
   - `scripts/docker/run_docker.sh` and `scripts/docker/run_docker.ps1` for host preflight and orchestration.

---

## 3. Verification & Test Coverage

- **`tests/inference/test_hardware_check.py`** (10 unit tests):
  - CPU-only detection and diagnostic failure when `require_gpu=True`.
  - Mock GPU detection with VRAM calculation and compute capability validation.
  - VRAM threshold calculation for default (16GB) vs. low-VRAM (11.5GB) modes.
  - Architecture warnings for compute capability < sm_75.
  - Config discovery for `.env.inference` and Critic python paths.
  - CLI execution via `scripts/inference/check_hardware.py --json`.
- **`tests/inference/test_docker_config.py`** (6 unit tests):
  - Validates `docker/Dockerfile.inference` base image, multi-venv creation, environment variables, and ports.
  - Validates `docker/Dockerfile.frontend` Node 20 base, port 3000, and server execution.
  - Validates `docker/entrypoint.sh` hardware check enforcement and fail-fast behavior.
  - Validates `docker-compose.yml` syntax, GPU device reservations, volumes, and `/healthz` healthcheck.
  - Validates `.dockerignore` excludes caches, zip archives, and local test artifacts.
  - Validates `run_docker.sh` and `run_docker.ps1` scripts exist and probe host container runtimes.

**Test Results**:
- `pytest tests/inference`: 185 passed, 0 failed.
- Monorepo full suite (`pytest tests/`): 481 passed, 1 skipped, 0 failed.
