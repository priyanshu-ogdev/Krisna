#!/usr/bin/env bash
# =============================================================================
# scripts/data-forge/setup_env.sh
# Linux & Windows WSL2 environment setup for Krisna Data-Forge.
#
# Optimized for:
#   - NVIDIA RTX A6000 (48GB VRAM) / CUDA 12.4+
#   - 128GB System RAM
#   - Full vLLM & FAISS-GPU support
#
# Usage (run from monorepo root or script directory):
#   ./scripts/data-forge/setup_env.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo -e "\033[1;36m================================================================\033[0m"
echo -e "\033[1;36m   Krisna Data-Forge — Linux / WSL2 Environment Setup           \033[0m"
echo -e "\033[1;36m================================================================\033[0m"

# 1. Check Python version
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo -e "\033[1;31mERROR: ${PYTHON_BIN} not found in PATH.\033[0m"
    echo -e "\033[1;33mInstall Python 3.10 or newer (e.g. sudo apt install python3-venv python3-pip)\033[0m"
    exit 1
fi

PY_VER="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo -e "\033[1;32mDetected Python: ${PYTHON_BIN} (${PY_VER})\033[0m"

# 2. Create or reuse virtual environment
VENV_DIR="${REPO_ROOT}/.venv-forge"
if [ ! -d "${VENV_DIR}" ]; then
    echo -e "\033[1;34mCreating virtual environment at ${VENV_DIR}...\033[0m"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools

# 3. Install PyTorch with CUDA 12.4 support
echo -e "\033[1;34mInstalling PyTorch with CUDA 12.4 support...\033[0m"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install FAISS-GPU (Linux x86_64)
echo -e "\033[1;34mInstalling FAISS-GPU...\033[0m"
pip install faiss-gpu || {
    echo -e "\033[1;33mfaiss-gpu pip install failed; falling back to faiss-cpu...\033[0m"
    pip install faiss-cpu
}

# 5. Install vLLM (Linux/WSL2 CUDA native engine)
echo -e "\033[1;34mInstalling vLLM...\033[0m"
pip install "vllm>=0.18.0"

# 6. Install Data-Forge package in editable mode with development dependencies
echo -e "\033[1;34mInstalling data-forge in editable mode...\033[0m"
pip install -e "./data-forge[dev]"

echo -e "\033[1;32m================================================================\033[0m"
echo -e "\033[1;32m   Data-Forge setup complete!                                   \033[0m"
echo -e "\033[1;32m================================================================\033[0m"
echo -e "\033[1;33mActivate environment: source ${VENV_DIR}/bin/activate\033[0m"
echo -e "\033[1;33mRun doctor check:     ./scripts/data-forge/run_data_forge.sh doctor\033[0m"
echo -e "\033[1;33mRun dry-run:          ./scripts/data-forge/run_data_forge.sh run --dry-run\033[0m"
