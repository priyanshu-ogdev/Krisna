#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting sketch tier training — Stage 1 (256px)...\033[0m"
source .venv/bin/activate
python -m krisna_training.sketch.train --config training/configs/sketch_train_stage1_256.yaml
