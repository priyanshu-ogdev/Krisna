#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/polish_qwen_image21_lora.yaml}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Detect and activate virtualenv if present
for venv_candidate in "$ROOT/.venv" "$ROOT/.venv-forge"; do
    if [ -f "$venv_candidate/bin/activate" ]; then
        source "$venv_candidate/bin/activate"
        break
    fi
done

PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
fi

export PYTHONPATH="$ROOT/training/src${PYTHONPATH:+:$PYTHONPATH}"

echo -e "\033[1;36mRunning Qwen-Image-2.1 training preflight and launcher...\033[0m"
# Preflight validates dataset receipts, split isolation, cryptographic hashes,
# diffusers pinned HEAD, and starts the accelerate trainer.
"${PYTHON_CMD}" -m krivex_training.polish.train_preflight "$CONFIG" "$@"

echo -e "\033[1;36mWriting dataset provenance & model manifest...\033[0m"
"${PYTHON_CMD}" - "$CONFIG" <<'PYEOF'
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
