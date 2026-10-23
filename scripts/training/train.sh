#!/usr/bin/env bash
# =============================================================================
# scripts/training/train.sh
# Training dispatcher for the Krivex Diffusion Polish Adaptation subsystem.
#
# Usage:
#   ./scripts/training/train.sh --list
#   ./scripts/training/train.sh audit
#   ./scripts/training/train.sh polish-default [config.yaml]
#   ./scripts/training/train.sh polish-dpo [config.yaml]
#   ./scripts/training/train.sh polish-cfm
#
# Tiers:
#   polish-default : Qwen/Qwen-Image-2.1 paired image-conditioned LoRA/DoRA
#   polish-dpo     : Continuous Flow Matching Direct Preference Optimization
#   polish-cfm     : Consistency Flow Matching distillation (~0.9s preview)
#   sketch-stage1  : PRODUCTION Deterministic Layout IR (krivex.layout.v1; 0 MB VRAM)
#   planner        : FROZEN Qwen3.5-9B GGUF + UICrit RAG (PRD §6; no training)
#   critic         : FROZEN Resident Review pass (PRD §6; no training)
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

list_tiers() {
    cat <<'EOF'
Tier            Status       Design intent (see docs/training/ARCHITECTURE.md)
--------------  -----------  ------------------------------------------------
sketch-stage1   PRODUCTION   Deterministic Layout IR engine (krivex.layout.v1; 0 MB VRAM, <5ms CPU).
sketch-stage2   RETIRED      Legacy LayoutFlow neural ablation deleted; production mainline is deterministic.
polish-default  PRODUCTION   Qwen/Qwen-Image-2.1 paired image-conditioned LoRA/DoRA (rank 64, alpha 32).
polish-dpo      PRODUCTION   Continuous Flow-Matching Direct Preference Optimization (beta=2000.0, anchor=0.1).
polish-cfm      PRODUCTION   Consistency Flow Matching distillation (4-step preview generator).
planner         FROZEN       Qwen3.5-9B GGUF + UICrit RAG; no Planner LoRA training (0 MB added VRAM).
critic          FROZEN       Resident Review pass; no Critic training in active pipeline.
EOF
}

case "${1:-}" in
    --list|"")
        list_tiers
        ;;
    audit)
        exec python3 "${REPO_ROOT}/scripts/training/audit_training_pipeline.py"
        ;;
    sketch-stage1|sketch-stage2)
        echo "Sketch tier is 100% deterministic Layout IR (krivex.layout.v1; 0 MB VRAM, <5ms CPU). No training occurs." >&2
        exit 0
        ;;
    polish-default)
        shift || true
        exec "${REPO_ROOT}/scripts/training/train_polish_default_lora.sh" "$@"
        ;;
    ablations)
        shift || true
        exec python3 -m krivex_training.polish.ablation_runner --all "$@"
        ;;
    polish-dpo)
        shift || true
        exec "${REPO_ROOT}/scripts/training/train_polish_dpo.sh" "$@"
        ;;
    polish-cfm)
        shift || true
        exec "${REPO_ROOT}/scripts/training/train_polish_fast_preview.sh" "$@"
        ;;
    planner)
        echo "Planner tier is 100% frozen (Qwen3.5-9B GGUF + UICrit RAG). No training occurs under PRD §6." >&2
        exit 0
        ;;
    critic)
        echo "Critic tier is 100% frozen (Resident Review pass). No training occurs under PRD §6." >&2
        exit 0
        ;;
    *)
        echo -e "\033[1;31mUnknown tier: $1\033[0m"
        list_tiers
        exit 1
        ;;
esac
