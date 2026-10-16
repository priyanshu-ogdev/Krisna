# Krisna training package

This package contains training-side data preparation, model primitives, and
launcher integrations. The current architecture and runbook live in:

- [`../docs/training/ARCHITECTURE.md`](../docs/training/ARCHITECTURE.md)
- [`../docs/training/TRAINING_RUNBOOK.md`](../docs/training/TRAINING_RUNBOOK.md)

## Current implementation status

| Area | Status | Description |
|---|---|---|
| `polish/prepare_dataset.py` | Active | Strict wireframe-conditioned target format dataset bridge with source-caption mixing |
| `polish/qwen_trainer_contract.py` | Active | Pinned Diffusers trainer contract for Qwen-Image-2.1 LoRA/DoRA |
| `polish/train_dpo.py` | Active | Phase 3 Flow-Matching DPO preference alignment (Wallace et al., Kim et al., Croitoru et al.) |
| `polish/cfm_loss.py` | Active | Phase 4 Consistency Flow Matching few-step distillation (Luo et al., Song et al.) |
| `data_forge_bridge/` | Active | Real-time sync adapters for Polish pairs (`sync_polish_default.py`) and preference pairs (`sync_dpo_pairs.py`) |
| `preference/` & `dpo/` | Active | Unified SQLite PreferenceStore with idempotent `INSERT OR IGNORE` and PRD §6 no-AI-judge constraint |
| `planner/` | Frozen | Frozen Qwen2.5-VL-7B-Instruct / Qwen3.5-9B GGUF + UICrit RAG; no Planner LoRA under PRD §6 |
| `critic/` | Frozen | Resident review pass within Planner tier (Gemma 4 31B external training retired for 12GB VRAM envelope) |
| `sketch/` & `layout/` | Deterministic | Deterministic Layout IR (`krisna.layout.v1`) baseline; discrete MaskGIT codebook training retired |

## Active model contract

Krisna's active training stack trains **one generative diffusion tier** across three progressive stages:

```text
Stage 1 (Supervised Base):
  condition_image (wireframe) + target_image (screenshot) + caption
      -> Pinned Diffusers (efe46a5b1...) + Qwen-Image-2.1 (d26bb6123...)
      -> DoRA / LoRA rank=64, alpha=32, random_flip=false
      -> Checkpoint: checkpoints/qwen_image21_lora/

Stage 2 (Preference Alignment):
  preference_pairs.db (2,200 human & verifier pairs) + Stage 1 LoRA
      -> Flow-Matching DPO (beta=2000, fm_anchor_weight=0.1)
      -> Checkpoint: checkpoints/polish_qwen_image21_dpo/

Stage 3 (Fast Preview Distillation):
  Consistency Flow Matching (CFM) distillation of Stage 1/2 teacher
      -> Pseudo-Huber consistency loss (huber_c=0.001)
      -> 4–8 step rapid ODE preview student (~0.9s on A6000, ~2.5s on RTX 4070)
      -> Checkpoint: checkpoints/qwen_image21_cfm_fast/
```

The Qwen base model remains frozen across all stages. The repository enforces receipt generation, validation holdout integrity, and multi-platform PowerShell and Bash launcher parity.

`polish_qwen_image21_lora.yaml` contains repository metadata as well as
Diffusers options. `qwen_trainer_contract.py` validates that metadata, maps the
pinned base revision to the official trainer's `--revision` option, and rejects
unsupported arguments before `accelerate` starts. The setup script checks out
Diffusers commit `efe46a5b177af67c64b3666da8ddddac7ee12163`; it does not track
the moving default branch.

## Read-only verification

```bash
python scripts/training/audit_training_pipeline.py
```

This command reports current model IDs, source/launcher availability, and
realized data counts without installing dependencies or starting training.
