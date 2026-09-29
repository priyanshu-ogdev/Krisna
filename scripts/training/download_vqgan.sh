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
