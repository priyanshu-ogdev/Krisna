#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/configs/polish_stage1_default_lora.yaml}"
[[ ! -f "$CONFIG" ]] && CONFIG="training/configs/polish_default_lora_z_image.yaml"

echo -e "\033[1;36mLaunching Z-Image LoRA training via diffusers' official script...\033[0m"
source .venv/bin/activate

python3 - "$CONFIG" <<'PYEOF'
import subprocess
import sys
import yaml
from pathlib import Path

config_path = sys.argv[1]
cfg = yaml.safe_load(Path(config_path).read_text())

diffusers_dir = Path(cfg.pop("diffusers_dir"))
script = diffusers_dir / "examples" / "dreambooth" / "train_dreambooth_lora_z_image.py"
if not script.exists():
    raise SystemExit(
        f"Script not found at {script} — run ./scripts/training/setup_env_diffusers_training.sh first."
    )

args = ["accelerate", "launch", str(script)]
for key, value in cfg.items():
    if value is None:
        continue
    if isinstance(value, bool):
        if value:
            args.append(f"--{key}")
        continue
    args.append(f"--{key}={value}")

print("Running:", " ".join(args))
subprocess.run(args, check=True)
PYEOF

echo -e "\033[1;32mDone. LoRA at the output_dir in $CONFIG.\033[0m"
echo -e "\033[1;33mUse it: export KRISNA_POLISH_DEFAULT_LORA_PATH=<output_dir>\033[0m"
