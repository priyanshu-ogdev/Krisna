#!/usr/bin/env bash
# =============================================================================
# run_data_forge.sh (Repository Root)
# Master launcher & orchestrator for the Krisna Data-Forge pipeline.
#
# Optimized for:
#   - NVIDIA RTX A6000 (48GB VRAM) / 128GB RAM / Core i9
#   - Linux / WSL2 Ubuntu (CUDA native vLLM & FAISS-GPU)
#
# Commands:
#   ./run_data_forge.sh doctor                  # Pre-flight environment & toolchain validation
#   ./run_data_forge.sh run [--dry-run]         # Run data pipeline (or dry run)
#   ./run_data_forge.sh run --stages 0,1,2      # Run specific stages
#   ./run_data_forge.sh manifest stats          # Inspect SQLite manifest metrics
#   ./run_data_forge.sh sync-training           # Bridge exported model_data into ./data/ for training
#   ./run_data_forge.sh all [--dry-run]         # Complete end-to-end: doctor -> run -> sync-training
#   ./run_data_forge.sh setup                   # Automated environment setup
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

COLOR_CYAN="\033[1;36m"
COLOR_GREEN="\033[1;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[1;31m"
COLOR_RESET="\033[0m"

echo -e "${COLOR_CYAN}================================================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}   Krisna Data-Forge — Master Pipeline Launcher                ${COLOR_RESET}"
echo -e "${COLOR_CYAN}================================================================${COLOR_RESET}"

# 1. Check or initialize .env
ENV_FILE="${REPO_ROOT}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo -e "${COLOR_YELLOW}[!] .env not found at repo root. Initializing from .env.example...${COLOR_RESET}"
    cp "${REPO_ROOT}/.env.example" "${ENV_FILE}"
    echo -e "${COLOR_YELLOW}[!] Please update .env with your Hugging Face HF_TOKEN before production runs.${COLOR_RESET}"
fi

# 2. Source .env variables
set -a
# shellcheck disable=SC1091
source "${ENV_FILE}"
set +a

# 3. Detect and activate Python virtual environment
VENV_ACTIVATED=0
for venv_candidate in "${REPO_ROOT}/.venv" "${REPO_ROOT}/.venv-forge"; do
    if [ -d "${venv_candidate}" ]; then
        if [ -f "${venv_candidate}/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "${venv_candidate}/bin/activate"
            VENV_ACTIVATED=1
            break
        fi
    fi
done

# 4. Check Python binary
if ! command -v python >/dev/null 2>&1; then
    echo -e "${COLOR_RED}ERROR: python not found in PATH.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Run './run_data_forge.sh setup' to configure the environment.${COLOR_RESET}"
    exit 1
fi

# 5. Set PYTHONPATH to include all monorepo src packages
export PYTHONPATH="${REPO_ROOT}/data-forge/src:${REPO_ROOT}/training/src:${REPO_ROOT}/inference/src:${REPO_ROOT}:${PYTHONPATH:-}"

# 6. Verify data_forge importability
if ! python -c "import data_forge" 2>/dev/null; then
    echo -e "${COLOR_YELLOW}[!] data_forge package not yet installed in current Python environment.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}[!] Installing data-forge in editable mode (-e ./data-forge[dev])...${COLOR_RESET}"
    pip install -e "./data-forge[dev]"
fi

# 7. Command Dispatcher
COMMAND="${1:-}"

case "${COMMAND}" in
    setup)
        echo -e "${COLOR_CYAN}[*] Running Linux/WSL2 automated environment setup...${COLOR_RESET}"
        exec ./scripts/data-forge/setup_env.sh
        ;;
    doctor)
        echo -e "${COLOR_CYAN}[*] Running pre-flight environment and toolchain validation...${COLOR_RESET}"
        exec python -m data_forge.cli doctor
        ;;
    run)
        shift
        echo -e "${COLOR_CYAN}[*] Launching Data-Forge pipeline...${COLOR_RESET}"
        python -m data_forge.cli run "$@"
        echo -e "${COLOR_GREEN}[✓] Data-Forge pipeline run finished.${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}Tip: To bridge exported data into ./data/ for model training, run:${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}     ./run_data_forge.sh sync-training${COLOR_RESET}"
        ;;
    manifest)
        shift
        exec python -m data_forge.cli manifest "$@"
        ;;
    registry)
        shift
        exec python -m data_forge.cli registry "$@"
        ;;
    sync-training)
        shift
        echo -e "${COLOR_CYAN}[*] Synchronizing exported data into training-ready datasets...${COLOR_RESET}"
        exec python "${REPO_ROOT}/scripts/data-forge/sync_to_training.py" "$@"
        ;;
    all)
        shift
        DRY_RUN_ARG=()
        for arg in "$@"; do
            if [ "$arg" = "--dry-run" ]; then
                DRY_RUN_ARG=("--dry-run")
            fi
        done

        echo -e "${COLOR_CYAN}[Step 1/3] Running doctor pre-flight check...${COLOR_RESET}"
        python -m data_forge.cli doctor || true

        echo -e "${COLOR_CYAN}[Step 2/3] Executing Data-Forge pipeline stages...${COLOR_RESET}"
        python -m data_forge.cli run "${DRY_RUN_ARG[@]}"

        echo -e "${COLOR_CYAN}[Step 3/3] Synchronizing datasets to model training layer...${COLOR_RESET}"
        python "${REPO_ROOT}/scripts/data-forge/sync_to_training.py"

        echo -e "${COLOR_GREEN}================================================================${COLOR_RESET}"
        echo -e "${COLOR_GREEN}   Complete Data-Forge Pipeline & Training Bridge Finished!     ${COLOR_RESET}"
        echo -e "${COLOR_GREEN}================================================================${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}Next: Start training using the unified dispatcher:${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./train.sh polish-default    # Z-Image-Turbo LoRA fine-tune${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./train.sh polish-dpo        # Diffusion-DPO preference alignment${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./train.sh sketch-stage1     # MaskGIT 256px from scratch${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./train.sh sketch-stage2     # MaskGIT 512px continuation${COLOR_RESET}"
        ;;
    *)
        echo -e "${COLOR_YELLOW}Usage: ./run_data_forge.sh <command> [options]${COLOR_RESET}"
        echo ""
        echo "Commands:"
        echo "  doctor         Run pre-flight checks (stage graph, toolchains, APIs)"
        echo "  run            Run the data pipeline (supports --dry-run, --resume, --stages)"
        echo "  manifest       Inspect SQLite database (stats, query, export)"
        echo "  registry       Check models and datasets upstream releases"
        echo "  sync-training  Bridge model_data/ into ./data/ ready for train.sh"
        echo "  all            Complete end-to-end flow: doctor -> run -> sync-training"
        echo "  setup          Set up virtual environment and install dependencies"
        echo ""
        echo "Examples:"
        echo "  ./run_data_forge.sh run --dry-run"
        echo "  ./run_data_forge.sh run --stages 0,1,2"
        echo "  ./run_data_forge.sh sync-training"
        echo "  ./run_data_forge.sh all"
        ;;
esac
