# scripts/training/

Unified training orchestrators, phase launchers, diagnostic auditors, and environment setup scripts for the **Krivex Diffusion Polish Adaptation** subsystem.

## Master Dispatchers (PowerShell & Bash Parity)

Training execution is managed across two 1:1 parity master orchestrators:

- **Windows PowerShell**: `.\scripts\training\train_all.ps1 [options]`
- **Linux / WSL2 Bash**: `./scripts/training/train.sh <command> [options]`

### Master Commands & Options

| Command / Option | Description | Target Phase |
| :--- | :--- | :--- |
| `audit` / `-Audit` | Read-only preflight audit of models, contracts, and dataset receipts | All |
| `polish-default` | Supervised Qwen-Image-2.1 DoRA base adaptation | **Phase 1 (Primary Target)** |
| `ablations` | Component ablation evaluation matrix (`ablation_runner.py`) | **Phase 2 (Scientific Matrix)** |
| `polish-dpo` | Continuous Flow-Matching Direct Preference Optimization | **Phase 3 (Preference Alignment)** |
| `polish-cfm` | Consistency Flow Matching fast preview distillation | **Phase 4 (Preview Distillation)** |
| `-Sync` / `sync-training` | Bridge exported Data-Forge records into training dataset format | Pre-Training Sync |
| `-DryRun` | Run CPU-only preflight verification and print accelerate launch command | All |
| `-List` / `--list` | Display tier trainability matrix and status | Reference |

---

## Quick Start Runbook

### Phase 1: Supervised Polish DoRA Training (Primary Server Target)

```powershell
# PowerShell (Windows)
# 1. Read-only pre-flight contract inspection
.\scripts\training\train_all.ps1 -Audit

# 2. CPU-only preflight check with command rendering
.\scripts\training\train_all.ps1 -Tier polish-default -DryRun

# 3. Live training launch
# Enterprise Max (48GB full utilization: batch=16, rank=128, FP32 AdamW, 14k steps, ~1.85h):
.\scripts\training\train_all.ps1 -Tier polish-default -Config training/configs/polish_qwen_image21_lora_a6000_max.yaml

# Enterprise Standard (48GB: batch=8, rank=64, 27.5k steps, ~15.2h):
.\scripts\training\train_all.ps1 -Tier polish-default

# Consumer Desktop (12GB NF4 QDoRA):
.\scripts\training\train_all.ps1 -Tier polish-default -Config training/configs/ablations/polish_qwen_qdora_12gb.yaml
```

```bash
# Bash (Linux / WSL2)
# 1. Read-only pre-flight contract inspection
./scripts/training/train.sh audit

# 2. Enterprise Max production run (14,000 steps; ~1.85h on A6000, 45GB peak):
./scripts/training/train.sh polish-default training/configs/polish_qwen_image21_lora_a6000_max.yaml \
  --max_train_steps 14000 --lr_warmup_steps 700

# 3. Enterprise Standard convergence run (27,500 steps; ~15.2h on A6000 Ada):
./scripts/training/train.sh polish-default training/configs/polish_qwen_image21_lora.yaml \
  --max_train_steps 27500 --lr_warmup_steps 500

# 4. Fast smoke test run (5,000 steps; ~40 min on A6000):
./scripts/training/train.sh polish-default
```

---

### Phase 2: Scientific Component Ablation Matrix

Runs the full ablation matrix across Attention vs MLP layers, +/- Wireframe, and +/- Layout RAG:

```powershell
# PowerShell
.\scripts\training\train_all.ps1 -Tier ablations
```

```bash
# Bash
./scripts/training/train.sh ablations
```

---

### Phase 3: Continuous Flow-Matching DPO Preference Alignment

> [!CAUTION]
> **Precondition**: Requires completed Phase 1 checkpoint (`checkpoints/polish_default_lora`). Auto-detected by `dpo_lora_autodetect.py` to prevent catastrophic forgetting.

```powershell
# PowerShell
.\scripts\training\train_all.ps1 -Tier polish-dpo
```

```bash
# Bash
./scripts/training/train.sh polish-dpo
```

---

### Phase 4: Consistency Flow Matching (CFM) Fast Preview Distillation

Distills 30-step teacher trajectory down to 4–6 steps ($<1.0\text{s}$ latency on A6000 Ada) for real-time preview:

```powershell
# PowerShell
.\scripts\training\train_all.ps1 -Tier polish-cfm
```

```bash
# Bash
./scripts/training/train.sh polish-cfm
```

---

## Detailed Script Catalog

### Orchestration & Preflight
- **`audit_training_pipeline.py`**: Read-only diagnostic checking model IDs, entry points, dataset pair counts, and hardware readiness.
- **`train_preflight.py`**: CPU-only preflight verifying dataset receipts, cryptographic image hashes, split isolation, pinned Diffusers commit, and building the exact `accelerate launch` command. Supports CLI parameter overrides (`--max_train_steps`, `--lr_warmup_steps`, etc.).
- **`write_qwen_image21_manifest.py`**: Writes cryptographic `manifest.json` beside checkpoint weights with base revision compatibility, dataset SHA-256 fingerprint, and split counts.

### Environment & Setup
- **`setup_env_diffusers_training.sh` / `.ps1`**: Clones pinned HuggingFace Diffusers at commit `efe46a5b177af67c64b3666da8ddddac7ee12163` and applies the native DoRA patch to `train_dreambooth_lora_qwenimage21_img2img.py`.
- **`prepare_polish_dataset.sh` / `.ps1`**: Converts exported Data-Forge pairs into typed HuggingFace dataset partitions.

### Optimization & Evaluation
- **`dpo_lora_autodetect.py`**: Automatically resolves and validates the base fine-tuned LoRA checkpoint path for DPO training continuation.
- **`compare_dpo_betas.py`**: Sweeps inverse-temperature $\beta \in [500, 2000, 5000]$ on validation pairs and writes `training_receipt.json`.
- **`compare_cfm_steps.py`**: Evaluates few-step consistency distillation trajectories ($N \in \{1, 2, 4, 6, 8\}$) and emits `distillation_receipt.json`.
- **`inspect_qwen_modules.py`**: Introspects Qwen-Image-2.1 transformer layers to verify target sublayer projections.

See [training/README.md](file:///d:/Krisna/training/README.md) for full architectural specifications and mathematical formulations.
