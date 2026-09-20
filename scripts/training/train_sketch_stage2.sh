#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting sketch tier training — Stage 2 (512px, progressive)...\033[0m"
echo -e "\033[1;33mRequires training/configs/sketch_train_stage2_512.yaml's init_from to point at a finished Stage 1 checkpoint.\033[0m"
source .venv/bin/activate
CONFIG="${1:-training/configs/sketch_stage2_512.yaml}"
[[ ! -f "$CONFIG" ]] && CONFIG="training/configs/sketch_train_stage2_512.yaml"
python -m krisna_training.sketch.train --config "$CONFIG"
echo -e "\033[1;32mDone. Checkpoint at checkpoints/sketch_stage2_512/checkpoint_final.pt\033[0m"
echo -e "\033[1;33mUse it: export KRISNA_SKETCH_CHECKPOINT=checkpoints/sketch_stage2_512/checkpoint_final.pt\033[0m"
echo -e "\033[1;33m(or just re-run scripts/inference/download_weights.py — it auto-discovers this path)\033[0m"
