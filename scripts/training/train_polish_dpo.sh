#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/training/train_polish_dpo.sh [config.yaml]

CONFIG="${1:-training/configs/polish_stage2_dpo_general.yaml}"
[[ ! -f "$CONFIG" ]] && CONFIG="training/configs/dpo_z_image_stage1_general.yaml"

echo -e "\033[1;36mLaunching Z-Image-Turbo Diffusion-DPO training...\033[0m"
echo -e "\033[1;33mSee docs/architecture/RESEARCH_AND_CITATIONS.md §4 for the loss\033[0m"
echo -e "\033[1;33mderivation, and training/README.md's 'DPO alignment' section.\033[0m"
source .venv/bin/activate

# BUG FIX (docs/review/27_dpo_base_lora_wiring.md): both general-DPO configs
# ship with `lora-adapter-path: null`, and this launcher used to translate
# the YAML straight to CLI args with nothing filling that gap in. Per
# train_dpo.py's own --lora-adapter-path help text, an unset value means
# "DPO trains a fresh adapter on top of the base pretrained weights" — so
# running the documented, canonical flow (./train.sh --all, or this script
# with its default config and no manual edit) silently threw away the
# polish-default stage's LoRA fine-tune entirely. The DPO stage would
# preference-align the vanilla pretrained Z-Image-Turbo, not the
# UI-domain-adapted one train.sh --list's own docstring describes
# ("...via diffusers' own dreambooth-LoRA script, THEN refined further by
# polish-dpo"), and the PRD's §6 model-stack table describes as ONE
# continuous "LoRA fine-tune + Diffusion-DPO" line, not two independent
# adapters.
#
# Fixed by auto-detecting the base fine-tune's checkpoint here (the same
# canonical output_dir both polish_stage1_default_lora.yaml and
# polish_default_lora_z_image.yaml use) when the config doesn't set
# lora-adapter-path itself, and loudly logging which path was taken either
# way — this must never be a silent fallback, matching this project's own
# "flag rather than silently degrade" discipline elsewhere (e.g.
# polish_quality_backend.py's `strength` kwarg, train_dpo.py's own
# encode_prompt() caveat).
BASE_LORA_OUTPUT_DIR="./checkpoints/polish_default_lora"

python3 - "$CONFIG" "$BASE_LORA_OUTPUT_DIR" <<'PYEOF'
import subprocess
import sys
from pathlib import Path

import yaml

# Import the standalone, unit-tested helper (see
# tests/training/test_dpo_lora_autodetect.py) rather than re-implementing
# the auto-detection logic inline in this heredoc. Assumes CWD is the
# monorepo root, same assumption this script's own header comment states.
sys.path.insert(0, "scripts/training")
from dpo_lora_autodetect import resolve_lora_adapter_path

config_path, base_lora_output_dir = sys.argv[1:3]
cfg = yaml.safe_load(Path(config_path).read_text())

cfg, message = resolve_lora_adapter_path(cfg, base_lora_output_dir)
print(message)

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
