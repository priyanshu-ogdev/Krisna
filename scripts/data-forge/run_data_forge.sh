#!/usr/bin/env bash
# =============================================================================
# scripts/data-forge/run_data_forge.sh
# Master launcher & orchestrator for the Krivex Data-Forge pipeline (Linux / WSL2).
#
# Commands:
#   ./scripts/data-forge/run_data_forge.sh doctor                  # Pre-flight environment validation
#   ./scripts/data-forge/run_data_forge.sh run [--dry-run]         # Run data pipeline (or dry run)
#   ./scripts/data-forge/run_data_forge.sh run --stages 0,1,2      # Run specific stages
#   ./scripts/data-forge/run_data_forge.sh manifest stats          # Inspect SQLite manifest metrics
#   ./scripts/data-forge/run_data_forge.sh sync-training           # Bridge exported model_data into training
#   ./scripts/data-forge/run_data_forge.sh all [--dry-run]         # Complete end-to-end flow
#   ./scripts/data-forge/run_data_forge.sh setup                   # Automated environment setup
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COLOR_CYAN="\033[1;36m"
COLOR_GREEN="\033[1;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[1;31m"
COLOR_RESET="\033[0m"

echo -e "${COLOR_CYAN}================================================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}   Krivex Data-Forge — Master Pipeline Launcher                ${COLOR_RESET}"
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

# 4. Check for python and data_forge
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "${COLOR_RED}[X] Python not found in environment.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Run './scripts/data-forge/run_data_forge.sh setup' to configure the environment.${COLOR_RESET}"
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
fi

export PYTHONPATH="${REPO_ROOT}/data-forge/src:${REPO_ROOT}/training/src:${REPO_ROOT}/inference/src:${REPO_ROOT}:${PYTHONPATH:-}"

COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    setup)
        echo -e "${COLOR_CYAN}[*] Running setup_env.sh...${COLOR_RESET}"
        exec "${REPO_ROOT}/scripts/data-forge/setup_env.sh" "$@"
        ;;
    doctor)
        echo -e "${COLOR_CYAN}[*] Running Data-Forge Pre-Flight Diagnostics...${COLOR_RESET}"
        "${PYTHON_CMD}" -m data_forge.cli doctor "$@"
        ;;
    run)
        echo -e "${COLOR_CYAN}[*] Executing Data-Forge Pipeline Engine...${COLOR_RESET}"
        "${PYTHON_CMD}" -m data_forge.cli run "$@"
        ;;
    manifest)
        echo -e "${COLOR_CYAN}[*] Querying Data-Forge Manifest SQLite Database...${COLOR_RESET}"
        "${PYTHON_CMD}" -m data_forge.cli manifest "$@"
        ;;
    audit|registry)
        echo -e "${COLOR_CYAN}[*] Auditing dataset & model registry preflight contracts...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/audit_registry.py" "$@"
        ;;
    pin-revisions)
        echo -e "${COLOR_CYAN}[*] Resolving unpinned revisions to immutable commit SHAs...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/pin_revisions.py" "$@"
        ;;
    verify-schemas)
        echo -e "${COLOR_CYAN}[*] Verifying JSON schemas against Pydantic definitions...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/verify_schemas.py" "$@"
        ;;
    sync-training)
        echo -e "${COLOR_CYAN}[*] Synchronizing processed model data into training directory...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/sync_to_training.py" "$@"
        ;;
    all)
        echo -e "${COLOR_CYAN}[Step 1/4] Pre-flight Environment Validation (doctor)...${COLOR_RESET}"
        "${PYTHON_CMD}" -m data_forge.cli doctor || echo -e "${COLOR_YELLOW}[!] Doctor reported warnings; continuing...${COLOR_RESET}"
        echo -e "${COLOR_CYAN}[Step 2/4] Running forensic registry audit...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/audit_registry.py"
        echo -e "${COLOR_CYAN}[Step 3/4] Pipeline Execution...${COLOR_RESET}"
        "${PYTHON_CMD}" -m data_forge.cli run "$@"
        echo -e "${COLOR_CYAN}[Step 4/4] Synchronizing Exported Datasets to Training Pool...${COLOR_RESET}"
        "${PYTHON_CMD}" "${REPO_ROOT}/scripts/data-forge/sync_to_training.py"
        echo -e "${COLOR_GREEN}================================================================${COLOR_RESET}"
        echo -e "${COLOR_GREEN}   Data-Forge Complete Workflow Finished Successfully!         ${COLOR_RESET}"
        echo -e "${COLOR_GREEN}================================================================${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}Ready for training:${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./scripts/training/train.sh polish-default${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}   ./scripts/training/train.sh polish-dpo${COLOR_RESET}"
        ;;
    help|--help|-h|*)
        echo -e "${COLOR_YELLOW}Usage: ./scripts/data-forge/run_data_forge.sh <command> [options]${COLOR_RESET}"
        echo ""
        echo "Commands:"
        echo "  doctor         Run pre-flight checks (stage graph, toolchains, APIs)"
        echo "  audit          Run forensic registry audit & contract validation"
        echo "  registry       Check models and datasets upstream releases / audit"
        echo "  pin-revisions  Resolve & pin model/dataset revisions to immutable git SHAs"
        echo "  verify-schemas Validate JSON schemas against Pydantic models"
        echo "  run            Run the data pipeline (supports --dry-run, --resume, --stages)"
        echo "  manifest       Inspect SQLite database (stats, query, export)"
        echo "  sync-training  Bridge model_data/ into training directory"
        echo "  all            Complete flow: doctor -> audit -> run -> sync-training"
        echo "  setup          Set up Linux/WSL2 environment"
        echo ""
        echo "Examples:"
        echo "  ./scripts/data-forge/run_data_forge.sh audit"
        echo "  ./scripts/data-forge/run_data_forge.sh pin-revisions --apply"
        echo "  ./scripts/data-forge/run_data_forge.sh run --dry-run"
        echo "  ./scripts/data-forge/run_data_forge.sh run --stages 0,1,2"
        echo "  ./scripts/data-forge/run_data_forge.sh sync-training"
        echo "  ./scripts/data-forge/run_data_forge.sh all"
        ;;
esac
