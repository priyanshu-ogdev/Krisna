#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${1:?Usage: prepare_sketch_dataset.sh <image_dir> <output_dir> [image_size] [captions.json]}"
OUTPUT_DIR="${2:?Usage: prepare_sketch_dataset.sh <image_dir> <output_dir> [image_size] [captions.json]}"
IMAGE_SIZE="${3:-256}"
CAPTIONS="${4:-}"
VQGAN_DIR="${VQGAN_DIR:-./checkpoints/vqgan}"

echo -e "\033[1;36mTokenizing $IMAGE_DIR -> $OUTPUT_DIR at ${IMAGE_SIZE}px...\033[0m"
source .venv/bin/activate

python3 - "$IMAGE_DIR" "$OUTPUT_DIR" "$IMAGE_SIZE" "$VQGAN_DIR" "$CAPTIONS" <<'PYEOF'
import sys
from krisna_training.sketch.vq_tokenizer import VQTokenizer
from krisna_training.sketch.prepare_dataset import prepare

image_dir, output_dir, image_size, vqgan_dir, captions = sys.argv[1:6]
tokenizer = VQTokenizer(
    checkpoint_path=f"{vqgan_dir}/last.ckpt",
    config_path=f"{vqgan_dir}/model.yaml",
)
prepare(
    image_dir=image_dir,
    output_dir=output_dir,
    tokenizer=tokenizer,
    image_size=int(image_size),
    captions_path=captions or None,
)
PYEOF

echo -e "\033[1;32mDone. Manifest at $OUTPUT_DIR/manifest.jsonl\033[0m"
