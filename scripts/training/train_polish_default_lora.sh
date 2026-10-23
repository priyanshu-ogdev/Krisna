#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/polish_qwen_image21_lora.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/training/src${PYTHONPATH:+:$PYTHONPATH}"

echo -e "\033[1;36mRunning CPU-only Qwen-Image-2.1 training preflight...\033[0m"
# This performs all receipt, split, path, image-hash, and pinned-HEAD checks
# before accelerate is started.  Its command rendering redacts hub_token.
python3 -m krivex_training.polish.train_preflight "$CONFIG"

python3 - "$CONFIG" <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt = json.loads((Path(config["dataset_name"]) / "dataset_receipt.json").read_text(encoding="utf-8"))
args = [
    sys.executable, "scripts/training/write_qwen_image21_manifest.py",
    config["output_dir"], "--dataset-dir", config["dataset_name"],
    "--base-revision", config["pretrained_model_revision"],
    "--resolution", str(config.get("resolution", 1024)),
    "--inference-steps", str(config.get("validation_num_inference_steps", 40)),
    "--rank", str(config["rank"]), "--lora-alpha", str(config["lora_alpha"]),
    "--lora-target-modules", config.get("lora_layers", ""),
    "--adaptation", "dora" if config.get("use_dora", True) else "lora",
    "--dataset-manifest-sha256", receipt["manifest_sha256"],
    "--split-counts-json", json.dumps(receipt["split_counts"], sort_keys=True),
]
subprocess.run(args, check=True)
PYEOF

echo -e "\033[1;32mDone. LoRA at the output_dir in $CONFIG.\033[0m"
echo -e "\033[1;33mUse it: export KRIVEX_POLISH_LORA_PATH=<output_dir>\033[0m"
