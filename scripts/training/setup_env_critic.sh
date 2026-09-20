#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[1;36mCreating isolated critic venv (Gemma 4 31B + Unsloth)...\033[0m"
echo -e "\033[1;33mThis is DELIBERATELY separate from .venv — see training/requirements-critic.txt\033[0m"
echo -e "\033[1;33mfor why (transformers version conflict with the planner tier).\033[0m"

python3 -m venv venv-critic
source venv-critic/bin/activate
pip install --upgrade pip
pip install -r training/requirements-critic.txt
# --no-deps: install krisna_training's own code (needed so
# train_critic_qlora.py can `import krisna_training.critic...`)
# WITHOUT pulling in its main dependencies (fastapi/pydantic/uvicorn/etc,
# from pyproject.toml) — venv-critic never needs those, and installing
# them would risk clobbering the exact transformers==5.5.0 pin this venv
# exists for. Only training/critic's own submodules (masking.py,
# dataset.py — both dependency-free besides what's already in
# training/requirements-critic.txt) actually get imported here.
pip install -e ./training --no-deps
deactivate

echo -e "\033[1;32mDone. Critic worker interpreter: ./venv-critic/bin/python\033[0m"
echo -e "\033[1;33mSet KRISNA_CRITIC_VENV_PYTHON if you put it somewhere else.\033[0m"
