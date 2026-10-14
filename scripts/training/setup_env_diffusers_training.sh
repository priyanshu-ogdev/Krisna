#!/usr/bin/env bash
set -euo pipefail

DIFFUSERS_DIR="${1:-./third_party/diffusers}"
DIFFUSERS_REVISION="efe46a5b177af67c64b3666da8ddddac7ee12163"

if [ ! -d "$DIFFUSERS_DIR" ]; then
    git clone https://github.com/huggingface/diffusers.git "$DIFFUSERS_DIR"
fi

# Never pull, reset, or clean an existing checkout.  Local trainer changes are
# allowed, but the commit beneath them must be the reproducibility pin.
HEAD="$(git -C "$DIFFUSERS_DIR" rev-parse HEAD 2>/dev/null || true)"
if [ "$HEAD" != "$DIFFUSERS_REVISION" ]; then
    if [ -n "$(git -C "$DIFFUSERS_DIR" status --porcelain --untracked-files=all)" ]; then
        echo "Refusing to change dirty Diffusers checkout at $DIFFUSERS_DIR (HEAD $HEAD; expected $DIFFUSERS_REVISION)." >&2
        exit 2
    fi
    git -C "$DIFFUSERS_DIR" fetch origin "$DIFFUSERS_REVISION"
    git -C "$DIFFUSERS_DIR" checkout --detach "$DIFFUSERS_REVISION"
fi

source .venv/bin/activate
pip install -e "$DIFFUSERS_DIR"
pip install -r "$DIFFUSERS_DIR/examples/dreambooth/requirements.txt"
# Keep the two setup surfaces on the same explicit training dependencies.
pip install "accelerate==1.10.1" "peft==0.17.1" "datasets==4.2.1" "pyarrow==21.0.0"

test -f "$DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"
echo "Pinned native Qwen-Image-2.1 trainer ready at $DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"
