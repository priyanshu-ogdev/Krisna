#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting sketch tier training — Stage 1 (256px)...\033[0m"
source .venv/bin/activate
CONFIG="${1:-training/configs/sketch_stage1_256.yaml}"
[[ ! -f "$CONFIG" ]] && CONFIG="training/configs/sketch_train_stage1_256.yaml"
python -m krisna_training.sketch.train --config "$CONFIG"
echo -e "\033[1;32mDone. Checkpoint at checkpoints/sketch_stage1_256/checkpoint_final.pt\033[0m"
echo -e "\033[1;33mNext: ./scripts/training/train_sketch_stage2.sh (set its yaml's init_from to this checkpoint)\033[0m"
