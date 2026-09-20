#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/setup_env.sh

echo -e "\033[1;36mSetting up Krisna Inference environment (Linux)...\033[0m"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Editable-installs both packages in the correct dependency order —
# krisna_training depends on krisna_inference (shared CLIP utility, see
# docs/architecture/DIRECTORY_LAYOUT.md), not the other way around, but a
# single dev environment commonly wants both available together.
pip install -e "./inference[dev]"
pip install -e "./training[dev]" --no-deps   # --no-deps: krisna-inference already satisfied above

echo -e "\033[1;32mEnvironment setup complete.\033[0m"
echo -e "\033[1;33mActivate later with: source .venv/bin/activate\033[0m"
echo -e "\033[1;33mRun tests with:      ./scripts/inference/run_tests.sh\033[0m"
echo -e "\033[1;33mStart the service:   ./scripts/inference/run_service.sh\033[0m"
