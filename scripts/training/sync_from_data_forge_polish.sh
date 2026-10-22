#!/usr/bin/env bash
set -euo pipefail

DATA_FORGE_MODEL_DATA="${1:?Usage: sync_from_data_forge_polish.sh <data_forge_model_data_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: sync_from_data_forge_polish.sh <data_forge_model_data_dir> <output_dir>}"

echo -e "\033[1;31mThe old single-image data-forge sync cannot produce Qwen condition/target pairs.\033[0m"
echo -e "\033[1;33mExport paired condition and target images, then use prepare_polish_dataset.sh.\033[0m"
exit 2
source .venv/bin/activate

python3 - "$DATA_FORGE_MODEL_DATA" "$OUTPUT_DIR" <<'PYEOF'
import sys
from krivex_training.data_forge_bridge.sync_polish_default import sync

model_data_dir, output_dir = sys.argv[1:3]
out = sync(model_data_dir, output_dir)
print(f"Prepared dataset at {out}")
PYEOF

echo -e "\033[1;32mDone. Point training/configs/polish_default_lora_z_image.yaml's instance_data_dir at $OUTPUT_DIR.\033[0m"
