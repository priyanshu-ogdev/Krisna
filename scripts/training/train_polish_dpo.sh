#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/training/train_polish_dpo.sh [config.yaml]

CONFIG="${1:-training/configs/polish_qwen_image21_dpo.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo -e "\033[1;31mError: Config $CONFIG not found.\033[0m"
    exit 1
fi

echo -e "\033[1;32mLaunching Qwen-Image-2.1 Flow-Matching DPO Training\033[0m"
echo -e "\033[1;34mConfig: $CONFIG\033[0m"

# Auto-detect base fine-tune checkpoint if lora-adapter-path is unset in config
BASE_LORA_OUTPUT_DIR="./checkpoints/polish_default_lora"

python3 - "$CONFIG" "$BASE_LORA_OUTPUT_DIR" <<'PYEOF'
import subprocess
import sys
from pathlib import Path

import yaml

# Import the standalone helper (see tests/training/test_dpo_lora_autodetect.py)
sys.path.insert(0, "scripts/training")
from dpo_lora_autodetect import resolve_lora_adapter_path

config_path, base_lora_output_dir = sys.argv[1:3]
cfg = yaml.safe_load(Path(config_path).read_text())

cfg, message = resolve_lora_adapter_path(cfg, base_lora_output_dir)
print(message)

# Filter out non-CLI keys (e.g. metadata/pipeline tags)
IGNORED_KEYS = {"pipeline_class", "conditioning_contract", "ablation_id", "ablation_hypothesis"}

args = ["accelerate", "launch", "-m", "krivex_training.polish.train_dpo"]
for key, value in cfg.items():
    if key in IGNORED_KEYS or value is None:
        continue
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            args.append(flag)
        continue
    if isinstance(value, list):
        for item in value:
            args.append(f"{flag}={item}")
        continue
    args.append(f"{flag}={value}")

print("Running:", " ".join(args))
subprocess.run(args, check=True)
PYEOF

echo -e "\033[1;32mDone. Checkpoint at the output-dir in $CONFIG.\033[0m"
echo -e "\033[1;33mUse it: export KRISNA_POLISH_DEFAULT_LORA_PATH=<output-dir>/final\033[0m"
