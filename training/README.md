# Krisna training package

This package contains training-side data preparation, model primitives, and
launcher integrations. The current architecture and runbook live in:

- [`../docs/training/ARCHITECTURE.md`](../docs/training/ARCHITECTURE.md)
- [`../docs/training/TRAINING_RUNBOOK.md`](../docs/training/TRAINING_RUNBOOK.md)

## Current implementation status

| Area | Status |
|---|---|
| `polish/prepare_dataset.py` | Implemented strict condition/target format bridge |
| Qwen-Image-2.1 trainer | External launcher contract only; trainer is not vendored or pinned in this checkout |
| `data_forge_bridge/` | Implemented sync adapters, subject to real Data-Forge outputs |
| `planner/` | Deprecated reference implementation; Planner is frozen at inference |
| `dpo/` | Preference-data preparation only in the active design; historical DPO trainer is not active |
| `layout/` | Model primitive and dataset schema; no complete learned training loop |
| `sketch/` | No complete source-backed learned training entry point in the current checkout |

## Active model contract

The intended trainable path is a supervised image-conditioned LoRA adapter for
`Qwen/Qwen-Image-2.1`:

```text
condition_image + target image + caption
    -> pinned external trainer
    -> Qwen-Image-2.1 LoRA adapter
    -> manifest and training receipt
```

The Qwen base model remains frozen. The repository must prove the external
trainer's condition-image and trainable-parameter behavior with a one-batch
smoke test before a full run.

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
