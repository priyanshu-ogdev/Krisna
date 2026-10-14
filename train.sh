#!/usr/bin/env bash
# Root-level training dispatcher. The current active intended path is the
# Qwen-Image-2.1 paired image-edit LoRA launcher. Historical Sketch/DPO paths
# are deliberately rejected rather than pretending their missing trainers are
# active. See docs/training/ARCHITECTURE.md.
#
# Usage:
#   ./train.sh --list
#   ./train.sh audit
#   ./train.sh polish-default [config.yaml]
#   ./train.sh polish-dpo    # deferred; exits without launching
#
# Prerequisites this script does NOT run for you (deployment-specific
# paths, run once per machine):
#   ./setup.sh base sketch polish     # or whichever tiers you need
#   ./scripts/training/download_vqgan.sh
#   ./scripts/training/sync_from_data_forge_sketch.sh <data_forge_model_data_dir> <out_dir>
#   ./scripts/training/sync_from_data_forge_dpo.sh    <data_forge_model_data_dir> <out_dir>
#   ./scripts/training/sync_from_data_forge_polish.sh <data_forge_model_data_dir> <out_dir>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

list_tiers() {
    cat <<'EOF'
Tier            Status       Design intent (see docs/training/ARCHITECTURE.md)
--------------  -----------  ------------------------------------------------
sketch-stage1   RESEARCH     Deterministic Layout IR baseline; no complete learned trainer.
sketch-stage2   RESEARCH     Future LayoutFlow/MaskGIT ablation only.
polish-default  PENDING      Qwen/Qwen-Image-2.1 paired image-conditioned LoRA;
                               external trainer must be pinned and smoke-tested.
polish-dpo      DEFERRED     Historical path, not part of the active pipeline.
planner         FROZEN       Qwen3.5-9B GGUF + UICrit RAG; no Planner LoRA.
critic          FROZEN       No Critic training in the active pipeline.
EOF
}

confirm_deprecated() {
    local tier="$1"
    echo -e "\033[1;31m$tier is DEPRECATED — the live inference stack cannot load its output.\033[0m"
    echo -e "\033[1;31mSee training/src/krisna_training/$tier/__init__.py's DeprecationWarning\033[0m"
    echo -e "\033[1;31mand docs/review/README.md for the full history. Proceed anyway? [y/N]\033[0m"
    read -r reply
    [ "$reply" = "y" ] || [ "$reply" = "Y" ] || exit 1
}

case "${1:-}" in
    --list|"")
        list_tiers
        ;;
    audit)
        exec python3 scripts/training/audit_training_pipeline.py
        ;;
    sketch-stage1)
        echo "Sketch training is research-only; no source-backed trainer exists in this checkout." >&2
        exit 2
        ;;
    sketch-stage2)
        echo "Sketch training is research-only; no source-backed trainer exists in this checkout." >&2
        exit 2
        ;;
    polish-default)
        exec ./scripts/training/train_polish_default_lora.sh "${2:-}"
        ;;
    polish-dpo)
        echo "polish-dpo is deferred and is not part of the active Qwen-Image-2.1 pipeline." >&2
        exit 2
        ;;
    planner)
        confirm_deprecated planner
        exec ./scripts/training/train_planner_lora.sh
        ;;
    critic)
        confirm_deprecated critic
        exec ./scripts/training/train_critic_qlora.sh
        ;;
    *)
        echo -e "\033[1;31mUnknown tier: $1\033[0m"
        list_tiers
        exit 1
        ;;
esac
