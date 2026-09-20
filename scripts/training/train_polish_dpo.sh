#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/training/train_polish_dpo.sh [config.yaml]

CONFIG="${1:-training/configs/polish_stage2_dpo_general.yaml}"
[[ ! -f "$CONFIG" ]] && CONFIG="training/configs/dpo_z_image_stage1_general.yaml"

echo -e "\033[1;36mLaunching Z-Image-Turbo Diffusion-DPO training...\033[0m"
echo -e "\033[1;33mSee docs/architecture/RESEARCH_AND_CITATIONS.md §4 for the loss\033[0m"
echo -e "\033[1;33mderivation, and training/README.md's 'DPO alignment' section.\033[0m"
source .venv/bin/activate

python3 - "$CONFIG" <<'PYEOF'
import subprocess
import sys
import yaml
from pathlib import Path

config_path = sys.argv[1]
cfg = yaml.safe_load(Path(config_path).read_text())

args = ["accelerate", "launch", "-m", "krisna_training.polish.train_dpo"]
for key, value in cfg.items():
    if value is None:
        continue
    if isinstance(value, bool):
        if value:
            args.append(f"--{key}")
        continue
    if isinstance(value, list):
        # --source is repeatable (argparse action="append") — a list
        # value needs one --key per element, not a single malformed
        # "--key=['a', 'b']" the way the simpler polish-default launcher's
        # flat-scalar translation would produce.
        for item in value:
            args.append(f"--{key}={item}")
        continue
    args.append(f"--{key}={value}")

print("Running:", " ".join(args))
subprocess.run(args, check=True)
PYEOF

echo -e "\033[1;32mDone. Checkpoint at the output-dir in $CONFIG.\033[0m"
echo -e "\033[1;33mUse it: export KRISNA_POLISH_DEFAULT_LORA_PATH=<output-dir>/final\033[0m"
