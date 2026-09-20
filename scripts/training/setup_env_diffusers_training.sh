#!/usr/bin/env bash
set -euo pipefail

DIFFUSERS_DIR="${1:-./third_party/diffusers}"

echo -e "\033[1;36mCloning diffusers (official training scripts live here)...\033[0m"
if [ ! -d "$DIFFUSERS_DIR" ]; then
    git clone https://github.com/huggingface/diffusers.git "$DIFFUSERS_DIR"
else
    echo "Already cloned at $DIFFUSERS_DIR, pulling latest..."
    git -C "$DIFFUSERS_DIR" pull
fi

source .venv/bin/activate
pip install -e "$DIFFUSERS_DIR"
pip install -r "$DIFFUSERS_DIR/examples/dreambooth/requirements.txt"
if [ -f "$DIFFUSERS_DIR/examples/dreambooth/requirements_z_image.txt" ]; then
    pip install -r "$DIFFUSERS_DIR/examples/dreambooth/requirements_z_image.txt"
fi
pip install accelerate peft

echo -e "\033[1;32mDone.\033[0m"
echo -e "\033[1;33mOfficial Z-Image LoRA script: $DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_z_image.py\033[0m"
echo -e "\033[1;33mNo official Qwen-Image-Edit training script exists yet — see\033[0m"
echo -e "\033[1;33mtraining/src/krisna_training/polish/__init__.py for real alternatives.\033[0m"
echo -e "\033[1;33mRun 'accelerate config' before your first training run if you haven't already.\033[0m"
