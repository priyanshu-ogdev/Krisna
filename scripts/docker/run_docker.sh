#!/usr/bin/env bash
# =============================================================================
# scripts/docker/run_docker.sh
# Host preflight check and launcher for Krisna Docker services
#
# Usage:
#   ./scripts/docker/run_docker.sh            # Launch inference + frontend
#   ./scripts/docker/run_docker.sh --build    # Build containers first
#   ./scripts/docker/run_docker.sh --low-vram # Launch with 12GB offloading
#   ./scripts/docker/run_docker.sh --down     # Stop services
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================="
echo "       Krisna Docker Deployment & Preflight Launcher             "
echo "================================================================="

# 1. Verify Docker installation
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or not in PATH."
    echo "Install Docker Engine or Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker daemon is not running. Please start Docker service."
    exit 1
fi

# 2. Check for docker compose
COMPOSE_CMD="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        echo "[ERROR] Neither 'docker compose' nor 'docker-compose' found."
        exit 1
    fi
fi

# 3. Handle arguments
BUILD_FLAG=""
DETACH_FLAG=""
ACTION="up"
export KRISNA_LOW_VRAM_MODE="${KRISNA_LOW_VRAM_MODE:-0}"

for arg in "$@"; do
    case "$arg" in
        --build)
            BUILD_FLAG="--build"
            ;;
        --detach|-d)
            DETACH_FLAG="-d"
            ;;
        --low-vram)
            export KRISNA_LOW_VRAM_MODE="1"
            echo "[i] Low-VRAM mode enabled (target: 12GB envelope)."
            ;;
        --down)
            ACTION="down"
            ;;
        --help|-h)
            echo "Usage: $0 [--build] [--detach|-d] [--low-vram] [--down]"
            exit 0
            ;;
    esac
done

if [ "$ACTION" = "down" ]; then
    echo "[+] Stopping Krisna Docker services..."
    $COMPOSE_CMD down
    exit 0
fi

# 4. Host GPU preflight check
echo "[+] Probing NVIDIA Container Toolkit on host..."
if docker info 2>/dev/null | grep -iq "nvidia"; then
    echo "[i] NVIDIA container runtime detected."
else
    echo "[WARNING] NVIDIA container runtime not detected in docker info."
    echo "If running on bare-metal Linux with GPU, install NVIDIA Container Toolkit:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "On Windows/macOS Docker Desktop, ensure WSL2 GPU acceleration is enabled."
fi

# 5. Launch containers
echo "[+] Starting services with $COMPOSE_CMD $ACTION $BUILD_FLAG $DETACH_FLAG..."
$COMPOSE_CMD up $BUILD_FLAG $DETACH_FLAG

echo "================================================================="
echo "Krisna services started."
echo "Inference API : http://localhost:8420 (health: http://localhost:8420/healthz)"
echo "Web Studio UI : http://localhost:3000"
echo "================================================================="
