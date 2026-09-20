#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${1:?Usage: prepare_polish_dataset.sh <image_dir> <output_dir> [captions.json]}"
OUTPUT_DIR="${2:?Usage: prepare_polish_dataset.sh <image_dir> <output_dir> [captions.json]}"
CAPTIONS="${3:-}"

echo -e "\033[1;36mPreparing polish-tier dataset: $IMAGE_DIR -> $OUTPUT_DIR...\033[0m"
source .venv/bin/activate

python3 - "$IMAGE_DIR" "$OUTPUT_DIR" "$CAPTIONS" <<'PYEOF'
import json
import sys
from pathlib import Path
from krisna_training.polish.dataset_prep import prepare, count_prepared

image_dir, output_dir, captions_path = sys.argv[1:4]
out = prepare(image_dir, output_dir, captions=captions_path or None, use_shared_instance_prompt="a UI design")
print(f"Prepared {count_prepared(out)} images at {out}")
PYEOF

echo -e "\033[1;32mDone.\033[0m"
