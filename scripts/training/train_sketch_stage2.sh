#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting sketch tier training — Stage 2 (512px, progressive)...\033[0m"
echo -e "\033[1;33mRequires training/configs/sketch_train_stage2_512.yaml's init_from to point at a finished Stage 1 checkpoint.\033[0m"
source .venv/bin/activate
python -m krisna_training.sketch.train --config training/configs/sketch_train_stage2_512.yaml
