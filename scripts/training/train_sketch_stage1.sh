#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting sketch tier training — Stage 1 (256px)...\033[0m"
source .venv/bin/activate
python -m krisna_training.sketch.train --config training/configs/sketch_train_stage1_256.yaml
echo -e "\033[1;32mDone. Checkpoint at checkpoints/sketch_stage1_256/checkpoint_final.pt\033[0m"
echo -e "\033[1;33mNext: ./scripts/training/train_sketch_stage2.sh (set its yaml's init_from to this checkpoint)\033[0m"
