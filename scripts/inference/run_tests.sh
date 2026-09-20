#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/run_tests.sh

echo -e "\033[1;36mRunning Krisna Inference test suite...\033[0m"
source .venv/bin/activate
python -m pytest tests/inference -v --tb=short "$@"
