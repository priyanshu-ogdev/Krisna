#!/usr/bin/env bash
# =============================================================================
# scripts/data-forge/run_data_forge.sh
# Core launcher for the Krisna Data-Forge automated pipeline.
#
# Usage:
#   ./scripts/data-forge/run_data_forge.sh doctor
#   ./scripts/data-forge/run_data_forge.sh run --dry-run
#   ./scripts/data-forge/run_data_forge.sh run --resume
#   ./scripts/data-forge/run_data_forge.sh run --stages 0,1,2
#   ./scripts/data-forge/run_data_forge.sh manifest stats
#   ./scripts/data-forge/run_data_forge.sh manifest query --status routed
# =============================================================================

set -euo pipefail

# Navigate to monorepo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Auto-load .env from repo root if present
if [ -f ".env" ]; then
    echo -e "\033[1;36mLoading environment variables from .env...\033[0m"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Detect virtual environment (.venv)
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source ".venv/bin/activate"
    elif [ -f ".venv/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source ".venv/Scripts/activate"
    fi
fi

# Ensure data-forge package is in PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/data-forge:${REPO_ROOT}/data-forge/src:${PYTHONPATH:-}"

# Check Python availability
if ! command -v python >/dev/null 2>&1; then
    echo -e "\033[1;31mERROR: python not found in PATH.\033[0m"
    exit 1
fi

# Verify data_forge importability
if ! python -c "import data_forge" 2>/dev/null; then
    echo -e "\033[1;31mERROR: data_forge package could not be imported.\033[0m"
    echo -e "\033[1;33mEnsure dependencies are installed: pip install -e ./data-forge\033[0m"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo -e "\033[1;33mNo command given — showing data-forge help:\033[0m"
    exec python -m data_forge.cli --help
fi

exec python -m data_forge.cli "$@"
