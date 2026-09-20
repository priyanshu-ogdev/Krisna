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
#   ./train.sh --all [--dry-run] [--log [path]] [--sync [DATA_ROOT]]
#   ./train.sh sketch-stage1 [--dry-run] [--log [path]] [--sync [DATA_ROOT]]
#   ./train.sh sketch-stage2 [--dry-run] [--log [path]] [--sync [DATA_ROOT]]
#   ./train.sh polish-default [config.yaml] [--dry-run] [--log [path]] [--sync [DATA_ROOT]]
#   ./train.sh polish-dpo [config.yaml] [--dry-run] [--log [path]] [--sync [DATA_ROOT]]
#   ./train.sh planner        # DEPRECATED — prompts to confirm
#   ./train.sh critic         # DEPRECATED — prompts to confirm
#
# Options:
#   --all         Run all 4 active tiers sequentially in dependency order
#   --sync [DIR]  Auto-run data-forge sync before training (default data_root: ./data_krisna)
#   --dry-run     Run pre-flight checks and display commands without executing
#   --log [PATH]  Tee output to file (default: logs/training/<tier>_<timestamp>.log)
#   --list, -l    Display status and architectural rationale of all model tiers
#   --help, -h    Show this help message
#
# Prerequisites this script checks or references:
#   ./setup.sh base sketch polish     # or whichever tiers you need
#   ./scripts/training/download_vqgan.sh
#   python scripts/data-forge/sync_to_training.py --data-root <data_root>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

list_tiers() {
    cat <<'EOF'
Tier            Status       Design intent (per docs/review/)
--------------  -----------  ------------------------------------------------
sketch-stage1   ACTIVE       MaskGIT transformer, from scratch, 256px/16x16
                              grid. batch=32, lr=3e-4, 1000-step warmup,
                              adam_beta1=0.9, adam_beta2=0.96 (Chang et al. 2022).
                              caption_mix_ratio=0.95 (Betker et al. 2023 —
                              5% short-caption exposure prevents overfitting
                              to dense VLM caption style), cfg_dropout=0.1
                              (Muse/Chang et al. 2023 — enables real CFG at
                              inference, see docs/review/17). num_workers=8.
                              No gradient checkpointing needed at this sequence length.

sketch-stage2   ACTIVE       Same architecture, continued from stage1 via
                              init_from, 512px/32x32 grid. batch=16,
                              lr=1.5e-4 (lower, continuation-style),
                              adam_beta1=0.9, adam_beta2=0.96, num_workers=6.
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
                              adaptation), beta=2000 (matches Wallace et al.'s
                              own reported baseline; lower values are plausible
                              for flow-matching adaptations but no sweep has
                              been run). Reference-model CPU-offloaded
                              (docs/review/18) to fit 48GB alongside the
                              trainable policy copy. Resolution 512 is confirmed
                              deliberate for compute efficiency.

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

preflight_check() {
    local tier="$1"
    local config="${2:-}"
    local failed=0

    case "$tier" in
        sketch-stage1)
            local manifest="./data/sketch_train_256/manifest.jsonl"
            if [[ ! -f "$manifest" ]]; then
                echo -e "\033[1;31m[PRE-FLIGHT FAILED] Manifest not found: $manifest\033[0m" >&2
                echo -e "\033[1;33m  Run data sync first:\033[0m" >&2
                echo -e "\033[1;33m    python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>\033[0m" >&2
                echo -e "\033[1;33m    or run with: ./train.sh --sync <DATA_ROOT> sketch-stage1\033[0m" >&2
                failed=1
            else
                echo -e "\033[1;32m[PRE-FLIGHT OK] Manifest found: $manifest\033[0m"
            fi
            ;;
        sketch-stage2)
            local manifest="./data/sketch_train_512/manifest.jsonl"
            local init_ckpt="./checkpoints/sketch_stage1_256/checkpoint_final.pt"
            if [[ ! -f "$manifest" ]]; then
                echo -e "\033[1;31m[PRE-FLIGHT FAILED] Manifest not found: $manifest\033[0m" >&2
                echo -e "\033[1;33m  Run data sync first:\033[0m" >&2
                echo -e "\033[1;33m    python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>\033[0m" >&2
                echo -e "\033[1;33m    or run with: ./train.sh --sync <DATA_ROOT> sketch-stage2\033[0m" >&2
                failed=1
            else
                echo -e "\033[1;32m[PRE-FLIGHT OK] Manifest found: $manifest\033[0m"
            fi
            if [[ ! -f "$init_ckpt" ]]; then
                if [[ "$RUN_ALL" -eq 1 ]]; then
                    echo -e "\033[1;33m[PRE-FLIGHT NOTE] Stage 1 checkpoint ($init_ckpt) will be created during Stage 1.\033[0m"
                else
                    echo -e "\033[1;33m[PRE-FLIGHT WARNING] Stage 1 checkpoint not found at $init_ckpt\033[0m" >&2
                    echo -e "\033[1;33m  Make sure Stage 1 has completed, or update init_from in config.\033[0m" >&2
                    if [[ "$DRY_RUN" -eq 0 ]]; then
                        failed=1
                    fi
                fi
            else
                echo -e "\033[1;32m[PRE-FLIGHT OK] Stage 1 checkpoint found: $init_ckpt\033[0m"
            fi
            ;;
        polish-default)
            local data_dir="./data/polish_default_train"
            if [[ ! -d "$data_dir" ]]; then
                echo -e "\033[1;31m[PRE-FLIGHT FAILED] Training data dir not found: $data_dir\033[0m" >&2
                echo -e "\033[1;33m  Run data sync first:\033[0m" >&2
                echo -e "\033[1;33m    python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>\033[0m" >&2
                echo -e "\033[1;33m    or run with: ./train.sh --sync <DATA_ROOT> polish-default\033[0m" >&2
                failed=1
            else
                echo -e "\033[1;32m[PRE-FLIGHT OK] Training data dir found: $data_dir\033[0m"
            fi
            ;;
        polish-dpo)
            local db_file="krisna_preference_pairs.db"
            if [[ ! -f "$db_file" ]]; then
                echo -e "\033[1;31m[PRE-FLIGHT FAILED] Preference database not found: $db_file\033[0m" >&2
                echo -e "\033[1;33m  Run DPO data sync first:\033[0m" >&2
                echo -e "\033[1;33m    python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>\033[0m" >&2
                echo -e "\033[1;33m    or run with: ./train.sh --sync <DATA_ROOT> polish-dpo\033[0m" >&2
                failed=1
            else
                echo -e "\033[1;32m[PRE-FLIGHT OK] Preference database found: $db_file\033[0m"
            fi
            ;;
    esac

    return $failed
}

run_command_with_logging() {
    local logfile="$1"
    shift
    local cmd=("$@")

    if [[ -n "$logfile" ]]; then
        mkdir -p "$(dirname "$logfile")"
        echo -e "\033[1;36m[Logging tee -> $logfile]\033[0m"
        # Run command and tee to logfile while preserving exit status
        "${cmd[@]}" 2>&1 | tee -a "$logfile"
        local status="${PIPESTATUS[0]}"
        return "$status"
    else
        "${cmd[@]}"
    fi
}

run_tier() {
    local tier="$1"
    local config="${2:-}"
    local logfile="${3:-}"

    echo -e "\n\033[1;36m============================================================\033[0m"
    echo -e "\033[1;36m  TIER: $tier\033[0m"
    echo -e "\033[1;36m============================================================\033[0m"

    if ! preflight_check "$tier" "$config"; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo -e "\033[1;33m[DRY RUN NOTE] Pre-flight would block a live run (see missing items above).\033[0m"
        else
            echo -e "\033[1;31mPre-flight check failed for $tier. Aborting.\033[0m" >&2
            return 1
        fi
    fi

    local cmd=()
    case "$tier" in
        sketch-stage1)
            cmd=("./scripts/training/train_sketch_stage1.sh")
            ;;
        sketch-stage2)
            cmd=("./scripts/training/train_sketch_stage2.sh")
            ;;
        polish-default)
            cmd=("./scripts/training/train_polish_default_lora.sh")
            [[ -n "$config" ]] && cmd+=("$config")
            ;;
        polish-dpo)
            cmd=("./scripts/training/train_polish_dpo.sh")
            [[ -n "$config" ]] && cmd+=("$config")
            ;;
        planner)
            confirm_deprecated planner
            cmd=("./scripts/training/train_planner_lora.sh")
            ;;
        critic)
            confirm_deprecated critic
            cmd=("./scripts/training/train_critic_qlora.sh")
            ;;
        *)
            echo -e "\033[1;31mUnknown tier: $tier\033[0m" >&2
            list_tiers
            return 1
            ;;
    esac

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo -e "\033[1;35m[DRY RUN] Would execute: ${cmd[*]}\033[0m"
        echo -e "\033[1;35m[DRY RUN] In directory:   $(pwd)\033[0m"
        [[ -n "$logfile" ]] && echo -e "\033[1;35m[DRY RUN] Log destination: $logfile\033[0m"
        return 0
    fi

    run_command_with_logging "$logfile" "${cmd[@]}"
}

# --- CLI Argument Parsing ---
DRY_RUN=0
RUN_ALL=0
DO_SYNC=0
SYNC_DATA_ROOT=""
LOG_FILE=""
ACTION=""
CONFIG_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list|-l)
            ACTION="list"
            shift
            ;;
        --help|-h)
            ACTION="help"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --all)
            RUN_ALL=1
            shift
            ;;
        --sync)
            DO_SYNC=1
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                SYNC_DATA_ROOT="$2"
                shift 2
            else
                shift
            fi
            ;;
        --sync=*)
            DO_SYNC=1
            SYNC_DATA_ROOT="${1#*=}"
            shift
            ;;
        --log)
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                LOG_FILE="$2"
                shift 2
            else
                LOG_FILE="auto"
                shift
            fi
            ;;
        --log=*)
            LOG_FILE="${1#*=}"
            shift
            ;;
        -*)
            echo -e "\033[1;31mUnknown option: $1\033[0m" >&2
            echo "Use ./train.sh --help for usage information."
            exit 1
            ;;
        *)
            if [[ -z "$ACTION" ]]; then
                ACTION="$1"
            elif [[ -z "$CONFIG_ARG" ]]; then
                CONFIG_ARG="$1"
            else
                echo -e "\033[1;31mUnexpected extra argument: $1\033[0m" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ "$ACTION" == "help" ]]; then
    head -n 25 "$0" | grep -E '^# ?' | sed 's/^# \?//'
    exit 0
fi

if [[ "$ACTION" == "list" || ( -z "$ACTION" && "$RUN_ALL" -eq 0 && "$DO_SYNC" -eq 0 ) ]]; then
    list_tiers
    exit 0
fi

# Run pre-training data sync if requested
if [[ "$DO_SYNC" -eq 1 ]]; then
    echo -e "\033[1;36m============================================================\033[0m"
    echo -e "\033[1;36m  RUNNING DATA-FORGE -> TRAINING DATA SYNC\033[0m"
    echo -e "\033[1;36m============================================================\033[0m"
    sync_cmd=("python" "scripts/data-forge/sync_to_training.py")
    if [[ -n "$SYNC_DATA_ROOT" ]]; then
        sync_cmd+=("--data-root" "$SYNC_DATA_ROOT")
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo -e "\033[1;35m[DRY RUN] Would execute: ${sync_cmd[*]}\033[0m"
    else
        "${sync_cmd[@]}"
    fi
    # If no tier or --all specified, exit after sync
    if [[ -z "$ACTION" && "$RUN_ALL" -eq 0 ]]; then
        echo -e "\033[1;32mSync completed successfully.\033[0m"
        exit 0
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Resolve log file if requested
if [[ "$LOG_FILE" == "auto" ]]; then
    mkdir -p logs/training
    if [[ "$RUN_ALL" -eq 1 ]]; then
        LOG_FILE="logs/training/all_stages_${TIMESTAMP}.log"
    else
        LOG_FILE="logs/training/${ACTION}_${TIMESTAMP}.log"
    fi
elif [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi

if [[ "$RUN_ALL" -eq 1 ]]; then
    stages=("sketch-stage1" "sketch-stage2" "polish-default" "polish-dpo")
    total=${#stages[@]}
    idx=1

    echo -e "\033[1;35m============================================================\033[0m"
    echo -e "\033[1;35m KRISNA FULL SEQUENTIAL TRAINING PIPELINE (${total} stages)\033[0m"
    [[ "$DRY_RUN" -eq 1 ]] && echo -e "\033[1;33m [DRY RUN MODE — NO COMMANDS WILL ACTUALLY BE EXECUTED]\033[0m"
    [[ -n "$LOG_FILE" ]] && echo -e "\033[1;36m Pipeline log destination: $LOG_FILE\033[0m"
    echo -e "\033[1;35m============================================================\033[0m"

    for stage in "${stages[@]}"; do
        echo -e "\n\033[1;34m>>> Stage ${idx}/${total}: ${stage} <<<\033[0m"
        run_tier "$stage" "" "$LOG_FILE"
        echo -e "\033[1;32m>>> Stage ${idx}/${total}: ${stage} COMPLETE <<<\033[0m"
        idx=$((idx + 1))
    done

    echo -e "\n\033[1;32m============================================================\033[0m"
    echo -e "\033[1;32m ALL ACTIVE TRAINING TIERS COMPLETED SUCCESSFULLY!\033[0m"
    echo -e "\033[1;32m============================================================\033[0m"
    exit 0
fi

run_tier "$ACTION" "$CONFIG_ARG" "$LOG_FILE"
