#!/usr/bin/env bash
set -euo pipefail

DATA_FORGE_MODEL_DATA="${1:?Usage: sync_from_data_forge_sketch.sh <data_forge_model_data_dir> <output_dir> [image_size]}"
OUTPUT_DIR="${2:?Usage: sync_from_data_forge_sketch.sh <data_forge_model_data_dir> <output_dir> [image_size]}"
IMAGE_SIZE="${3:-256}"
VQGAN_DIR="${VQGAN_DIR:-./checkpoints/vqgan}"

echo -e "\033[1;36mSyncing sketch tier data from data-forge ($DATA_FORGE_MODEL_DATA) -> $OUTPUT_DIR...\033[0m"
echo -e "\033[1;33mRe-tokenizing via boris/vqgan_f16_16384 (not data-forge's Open-MAGVIT2 .pt files) — see\033[0m"
echo -e "\033[1;33mtraining/data_forge_bridge/__init__.py for why.\033[0m"
source .venv/bin/activate

python3 - "$DATA_FORGE_MODEL_DATA" "$OUTPUT_DIR" "$IMAGE_SIZE" "$VQGAN_DIR" <<'PYEOF'
import sys
from krisna_training.sketch.vq_tokenizer import VQTokenizer
from krisna_training.data_forge_bridge.sync_sketch_tier import sync

model_data_dir, output_dir, image_size, vqgan_dir = sys.argv[1:5]
tokenizer = VQTokenizer(
    checkpoint_path=f"{vqgan_dir}/last.ckpt",
    config_path=f"{vqgan_dir}/model.yaml",
)
manifest_path = sync(model_data_dir, output_dir, tokenizer, image_size=int(image_size))
print(f"Manifest written to {manifest_path}")
PYEOF

echo -e "\033[1;32mDone.\033[0m"
