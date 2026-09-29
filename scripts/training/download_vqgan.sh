#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./checkpoints/vqgan}"
mkdir -p "$DEST"

echo -e "\033[1;36mDownloading boris/vqgan_f16_16384 (real, public VQGAN checkpoint)...\033[0m"
curl -L -o "$DEST/model.yaml" -C - \
  "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/configs/model.yaml&dl=1"
curl -L -o "$DEST/last.ckpt" -C - \
  "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/ckpts/last.ckpt&dl=1"

echo -e "\033[1;32mDone. Checkpoint at $DEST/last.ckpt, config at $DEST/model.yaml\033[0m"
echo -e "\033[1;33mNeeded by BOTH data-forge's Sketch-tier encoding AND inference's VQ-decode handoff (Finalize) —\033[0m"
echo -e "\033[1;33mnot a training-only file. For inference: export KRISNA_VQGAN_CHECKPOINT=$DEST/last.ckpt KRISNA_VQGAN_CONFIG=$DEST/model.yaml\033[0m"
echo -e "\033[1;33m(or just run scripts/inference/download_weights.py — it downloads this automatically if missing and writes those two vars for you)\033[0m"
