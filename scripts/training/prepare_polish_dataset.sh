#!/usr/bin/env bash
set -euo pipefail

CONDITION_DIR="${1:?Usage: prepare_polish_dataset.sh <condition_dir> <target_dir> <output_dir> [captions.json]}"
TARGET_DIR="${2:?Usage: prepare_polish_dataset.sh <condition_dir> <target_dir> <output_dir> [captions.json]}"
OUTPUT_DIR="${3:?Usage: prepare_polish_dataset.sh <condition_dir> <target_dir> <output_dir> [captions.json]}"
CAPTIONS="${4:-}"

echo -e "\033[1;36mPreparing Qwen-Image-2.1 pairs: $CONDITION_DIR + $TARGET_DIR -> $OUTPUT_DIR...\033[0m"
source .venv/bin/activate

python3 - "$CONDITION_DIR" "$TARGET_DIR" "$OUTPUT_DIR" "$CAPTIONS" <<'PYEOF'
import json
import sys
from pathlib import Path
from krivex_training.polish.prepare_dataset import prepare_image_pairs

condition_dir, target_dir, output_dir, captions_path = sys.argv[1:5]
captions = json.loads(Path(captions_path).read_text()) if captions_path else {}

out = prepare_image_pairs(condition_dir, target_dir, output_dir, captions=captions)
print(f"Prepared Qwen image pairs at {out}")
PYEOF

echo -e "\033[1;32mDone.\033[0m"
