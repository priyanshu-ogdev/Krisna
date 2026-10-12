#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[1;31m=========================================================\033[0m"
echo -e "\033[1;31mDEPRECATED — the final no-RLHF-loop PRD revision ships the\033[0m"
echo -e "\033[1;31mPlanner (Qwen3.5-9B) fully frozen. Design-domain grounding\033[0m"
echo -e "\033[1;31mcomes from RAG over data-forge's UICrit corpus instead. This\033[0m"
echo -e "\033[1;31mscript trains a real LoRA adapter, but inference/planner_backend.py\033[0m"
echo -e "\033[1;31mno longer has a lora_adapter_path parameter to load it with — see\033[0m"
echo -e "\033[1;31mtraining/src/krisna_training/planner/__init__.py and\033[0m"
echo -e "\033[1;31mdocs/review/03_planner.md for the full history.\033[0m"
echo -e "\033[1;31m=========================================================\033[0m"

echo -e "\033[1;31mRefusing to start: Planner training is removed from the production design.\033[0m"
echo -e "\033[1;33mUse scripts/inference/download_planner_models.py and evaluate_planner.py instead.\033[0m"
exit 2
