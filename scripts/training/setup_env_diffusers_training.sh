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

# Idempotently inject native DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. ICML 2024)
# into the pinned Qwen DreamBooth img2img trainer script.
python3 -c "
import sys
from pathlib import Path

script_path = Path('$DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py')
if not script_path.is_file():
    sys.exit(0)
content = script_path.read_text(encoding='utf-8')
if '--use_dora' not in content:
    arg_target = 'parser.add_argument(\"--lora_dropout\", type=float, default=0.0, help=\"Dropout probability for LoRA layers\")'
    dora_arg = '\n    parser.add_argument(\n        \"--use_dora\",\n        action=\"store_true\",\n        default=False,\n        help=\"Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation) instead of standard LoRA.\",\n    )'
    if arg_target in content:
        content = content.replace(arg_target, arg_target + dora_arg)
    else:
        sys.exit('Missing parser lora_dropout anchor')

    lora_target = 'init_lora_weights=\"gaussian\",\n        target_modules=target_modules,'
    dora_lora = 'init_lora_weights=\"gaussian\",\n        target_modules=target_modules,\n        use_dora=args.use_dora,'
    if lora_target in content:
        content = content.replace(lora_target, dora_lora)
    else:
        sys.exit('Missing LoraConfig target_modules anchor')

    script_path.write_text(content, encoding='utf-8')
    print('Successfully applied native DoRA patch to ' + str(script_path))
else:
    print('Native DoRA patch already present in ' + str(script_path))
"

echo "Pinned native Qwen-Image-2.1 trainer ready at $DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py"

