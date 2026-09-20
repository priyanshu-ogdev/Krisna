#!/usr/bin/env bash
set -euo pipefail

DATA_FORGE_MODEL_DATA="${1:?Usage: sync_from_data_forge_polish.sh <data_forge_model_data_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: sync_from_data_forge_polish.sh <data_forge_model_data_dir> <output_dir>}"

echo -e "\033[1;36mSyncing Z-Image-Turbo data from data-forge ($DATA_FORGE_MODEL_DATA) -> $OUTPUT_DIR...\033[0m"
source .venv/bin/activate

python3 - "$DATA_FORGE_MODEL_DATA" "$OUTPUT_DIR" <<'PYEOF'
import sys
from krisna_training.data_forge_bridge.sync_polish_default import sync

model_data_dir, output_dir = sys.argv[1:3]
out = sync(model_data_dir, output_dir)
print(f"Prepared dataset at {out}")
PYEOF

echo -e "\033[1;32mDone. Point training/configs/polish_default_lora_z_image.yaml's instance_data_dir at $OUTPUT_DIR.\033[0m"
