#!/usr/bin/env bash
# =============================================================================
# Krivex Inference Container Entrypoint
# Validates GPU & CUDA compatibility and model configuration before launch
# =============================================================================
set -e

echo "================================================================="
echo "       Starting Krivex Inference Service (Docker Container)       "
echo "================================================================="

# Activate main virtualenv
source /opt/venv-inference/bin/activate

# Source .env.inference if mounted or present
if [ -f "/app/.env.inference" ]; then
    echo "[i] Sourcing /app/.env.inference configuration..."
    set -a
    source /app/.env.inference
    set +a
fi

# Ensure data directories exist
mkdir -p /app/data/blobs /app/models /app/checkpoints

# Preflight hardware check
echo "[+] Running preflight hardware and GPU/CUDA verification..."
if [ "${KRIVEX_USE_REAL_BACKENDS:-1}" = "1" ]; then
    if ! python /app/scripts/inference/check_hardware.py --require-gpu; then
        echo ""
        echo "================================================================="
        echo "[FATAL ERROR] Container hardware preflight check FAILED."
        echo "Real inference backends (KRIVEX_USE_REAL_BACKENDS=1) require a"
        echo "compatible NVIDIA GPU and NVIDIA Container Toolkit on the host."
        echo ""
        echo "Host Remediation:"
        echo "1. Verify NVIDIA Container Toolkit is installed on the host:"
        echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        echo "2. Launch container with GPU access enabled:"
        echo "   docker run --gpus all -p 8420:8420 krivex-inference:latest"
        echo "3. If testing in CPU-only / mock mode explicitly:"
        echo "   Set KRIVEX_USE_REAL_BACKENDS=0 in environment"
        echo "================================================================="
        exit 1
    fi
else
    echo "[i] Running in MOCK / TEST mode (KRIVEX_USE_REAL_BACKENDS=0). Skipping hardware enforcement."
    python /app/scripts/inference/check_hardware.py || true
fi

echo "[+] Preflight checks passed. Launching inference service..."
exec "$@"
