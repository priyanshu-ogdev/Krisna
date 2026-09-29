#!/usr/bin/env bash
# Root-level data-forge pipeline launcher. Thin wrapper around the real
# `data-forge` console command (registered by data-forge/pyproject.toml's
# [project.scripts]) — that command IS the source of truth for the CLI
# surface (see data_forge/cli.py's own module docstring for the full
# `run`/`manifest query` usage), this just gives a stable root-level
# entry point that activates the right venv first.
#
# Usage:
#   ./run_data_forge.sh run --dry-run
#   ./run_data_forge.sh run --resume
#   ./run_data_forge.sh run --stages 0,1,2
#   ./run_data_forge.sh manifest query --status routed

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d ".venv" ]; then
    echo -e "\033[1;31m.venv not found — run ./setup.sh base first.\033[0m"
    exit 1
fi

source .venv/bin/activate

if ! python -c "import data_forge" 2>/dev/null; then
    echo -e "\033[1;31mkrisna-data-forge isn't installed in .venv yet.\033[0m"
    echo -e "\033[1;33mRun: pip install -e \"./data-forge\" (see ./setup.sh's [base] phase note\033[0m"
    echo -e "\033[1;33mabout data-forge's own heavier deps — install [dev] too if you need them).\033[0m"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo -e "\033[1;33mNo command given — showing data-forge's own usage:\033[0m"
    exec data-forge --help
fi

exec data-forge "$@"
