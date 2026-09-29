#!/usr/bin/env bash
# Root-level training dispatcher. Routes to the existing tier-specific
# scripts/training/train_*.sh — kept as the source of truth for actual
# invocation args (they translate each tier's real YAML config into the
# correct CLI flags; see training/configs/*.yaml for what those flags
# actually are and docs/review/18_training_memory_audit.md for the
# A6000 48GB/128GB-RAM sizing behind each tier's current settings).
#
# Usage:
#   ./train.sh --list
#   ./train.sh sketch-stage1
#   ./train.sh sketch-stage2
#   ./train.sh polish-default [config.yaml]
#   ./train.sh polish-dpo [config.yaml]
#   ./train.sh planner        # DEPRECATED — see below, will prompt to confirm
#   ./train.sh critic         # DEPRECATED — see below, will prompt to confirm
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
Tier            Status       Design intent (per docs/review/)
--------------  -----------  ------------------------------------------------
sketch-stage1   ACTIVE       MaskGIT transformer, from scratch, 256px/16x16
                              grid. batch=32, lr=3e-4, 1000-step warmup,
                              caption_mix_ratio=0.95 (Betker et al. 2023 —
                              5% short-caption exposure prevents overfitting
                              to dense VLM caption style), cfg_dropout=0.1
                              (Muse/Chang et al. 2023 — enables real CFG at
                              inference, see docs/review/17). No gradient
                              checkpointing needed at this sequence length.

sketch-stage2   ACTIVE       Same architecture, continued from stage1 via
                              init_from, 512px/32x32 grid. batch=16,
                              lr=1.5e-4 (lower, continuation-style),
                              use_gradient_checkpointing=true — REQUIRED
                              safeguard at 1024 tokens/seq, see
                              docs/review/18 for the real activation-memory
                              math behind this.

polish-default  ACTIVE       Z-Image-Turbo LoRA (rank=32; no separate alpha
                              field — diffusers' own dreambooth-LoRA
                              scripts default alpha=rank, confirmed as a
                              deliberate ecosystem-wide convention across
                              SDXL/SD3/vanilla dreambooth, see
                              docs/review/18) via diffusers' own
                              dreambooth-LoRA script, THEN refined further
                              by polish-dpo. Base fine-tune step.

polish-dpo      ACTIVE       Flow-matching-adapted Diffusion-DPO (Wallace
                              et al. 2024 + MotionFlux's velocity-prediction
                              adaptation), beta=2000 (Wallace et al.'s own
                              SD1.5 setting — but see dpo_loss.py's docstring
                              for newer, flow-matching-specific evidence
                              suggesting lower values, e.g. 500, may work
                              better here; no Z-Image-Turbo-specific sweep
                              has been run). Reference-model CPU-offloaded
                              (docs/review/18) to fit 48GB alongside the
                              trainable policy copy.

planner         DEPRECATED   Frozen at inference under the final no-RLHF-
                              loop PRD revision (RAG over UICrit corpus
                              instead) — training this produces an adapter
                              inference/planner_backend.py cannot load.
                              Kept as reference code only.

critic          DEPRECATED   Frozen, zero-shot at inference under the same
                              PRD revision. Same situation as planner —
                              training this produces an unused adapter.
                              Kept as reference code only.
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
    sketch-stage1)
        exec ./scripts/training/train_sketch_stage1.sh
        ;;
    sketch-stage2)
        exec ./scripts/training/train_sketch_stage2.sh
        ;;
    polish-default)
        exec ./scripts/training/train_polish_default_lora.sh "${2:-}"
        ;;
    polish-dpo)
        exec ./scripts/training/train_polish_dpo.sh "${2:-}"
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
