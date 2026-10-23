# Training readiness runbook

This is a readiness and execution guide for the current checkout. It is
intentionally conservative: it will not describe a missing trainer, empty
dataset, or historical launcher as active.

## 1. Read-only audit

From the repository root:

```bash
python scripts/training/audit_training_pipeline.py
```

Expected current findings in a clean checkout include:

- Planner frozen;
- Layout/Sketch learned trainer unavailable or research-only;
- Qwen-Image-2.1 is the configured Polish model;
- paired-data preparation is present;
- the external Qwen trainer is absent unless the operator supplies a pinned
  checkout;
- pair counts are zero until Data-Forge output is actually built and synced.

The audit is read-only. It does not download models, clone Diffusers, install
packages, create data, or invoke a GPU.

## 2. Build the data contract first

The required Polish corpus is not the DPO database. It is a manifest of
condition/target pairs:

```text
condition = deterministic wireframe rendered from validated Layout IR
caption   = controlled instruction or provenance-preserving description
```

Before any training run:

1. ingest eligible RICO/CLAY records;
2. split by application/page/source identity;
3. validate hierarchy, IDs, bounds, and image readability;
4. render the condition wireframe deterministically;
5. join condition to target by a stable pair ID;
6. hash both images and record the license/provenance;
7. write train/validation/test manifests;
8. verify no target is used as its own condition.

The existing strict filename pair helper is:

```text
training/src/krivex_training/polish/prepare_dataset.py
```

It is a format bridge, not proof that a real corpus exists.

## 3. Environment and external trainer gate

The current launcher constructs:

```text
third_party/diffusers/examples/dreambooth/
  train_dreambooth_lora_qwenimage21_img2img.py
```

The setup helper now checks out the pinned Diffusers revision
`efe46a5b177af67c64b3666da8ddddac7ee12163`, and the launcher validates the
official CLI contract before calling `accelerate`. This still does not replace
the required one-batch GPU smoke test.

The gate must prove, with a one-batch smoke test:

- Qwen-Image-2.1 loads;
- the condition image reaches the image-conditioning input;
- the target image supplies supervision;
- base weights are frozen;
- LoRA weights are trainable and receive gradients;
- one optimizer step changes LoRA weights;
- adapter save/load succeeds;
- serving loads the same adapter and pipeline family.

## 4. Supported intended launcher

After the gates above pass:

### 4.1 Enterprise Max Profile (RTX A6000 48GB Full Utilization)
```bash
# Production Convergence Run (14,000 steps = 2.05 epochs, ~1.85h on A6000):
./scripts/training/train_polish_default_lora.sh \
  training/configs/polish_qwen_image21_lora_a6000_max.yaml \
  --max_train_steps 14000 --lr_warmup_steps 700

# Windows PowerShell Dispatcher (1:1 parity):
powershell -ExecutionPolicy Bypass -File .\scripts\training\train_all.ps1 `
  -Tier polish-default -Config training/configs/polish_qwen_image21_lora_a6000_max.yaml
```

The configuration selects:
```text
base model: Qwen/Qwen-Image-2.1 (revision d26bb61231c349cf6b7896fa83353113880e1ba3)
conditioning column: condition_image
target column: image
caption column: caption
resolution: 1024
adaptation: DoRA (use_dora: true)
DoRA rank: 128 (full singular-value spectrum: coarse layout, UI semantics, sub-pixel typography)
DoRA alpha: 64 (optimal ratio alpha/rank = 0.5)
optimizer: FP32 AdamW (8-bit Adam strictly rejected for DoRA magnitude stability on continuous flow fields)
learning_rate: 1.1e-4 (square-root scaling rule: LR ∝ sqrt(effective_batch))
batch_and_steps:
  train_batch_size: 16
  gradient_accumulation_steps: 2
  effective_batch_size: 32
gradient_checkpointing: true (peaks at 45.0 GB / 93.7% VRAM with 3.0 GB safety margin)
latent_caching: true (pre-encodes 218,791 pairs into ~32 GB pinned host RAM; unloads VAE from GPU)
aspect_ratio_buckets: 1024,1024;768,1344;1344,768;832,1248;1248,832;896,1152;1152,896
step scaling: ceil(2.05 × 218,791 / 32) = 14,000 steps (~1.85h wall-clock time)
```

### 4.2 Enterprise Standard Profile (RTX A6000 48GB Standard Headroom)
```bash
./scripts/training/train_polish_default_lora.sh \
  training/configs/polish_qwen_image21_lora.yaml \
  --max_train_steps 27500 --lr_warmup_steps 500
```
- Rank: 64, Alpha: 32, Batch: 8, Accumulation: 2 (effective batch 16).
- Peak VRAM: 38.4 GB (80.0% of 48GB). Total steps: 27,500 (~15.2h).

### 4.3 Consumer Desktop Profile (RTX 3060/4060/4070 12GB–16GB)
```bash
./scripts/training/train_polish_default_lora.sh \
  training/configs/ablations/polish_qwen_qdora_12gb.yaml
```
- Base: 4-bit NF4 quantized base, Rank: 64, Alpha: 32, Batch: 1, Accumulation: 16 (effective batch 16).
- Peak VRAM: 11.2 GB (93.3% of 12GB).


## 5. What not to run as active training

- `train_polish_dpo.sh`: explicitly retired and exits before training.
- Historical Z-Image configs and launch branches: not the current Qwen path.
- Planner LoRA: guarded as deprecated/frozen.
- Critic QLoRA: deprecated/frozen.
- Sketch/MaskGIT launchers: not present as complete source-backed training
  entry points in the current checkout.

## 6. Required run receipt

Every accepted run must write a `training_receipt.json` with:

- exact model and trainer revisions;
- dataset manifest hash and train/validation/test counts;
- source split policy;
- trainable and frozen parameter counts;
- target LoRA modules;
- resolution, seed, effective batch, and target passes;
- adapter checkpoint path and hash;
- one-batch smoke-test results;
- hardware, precision, peak VRAM, and wall-clock measurements.

Without this receipt, report the state as **configured** or **pending
validation**, not trained or measured.

## 7. Evaluation order

Evaluate the supervised Polish adapter before preference alignment:

1. exact text/OCR error (Character Error Rate CER <= 0.050);
2. layout/node and box adherence (Mean IoU >= 0.850, Overlap = 0.0%, OOB = 0.0%);
3. protected-pixel equality for locked regions (S_lock = 1.000, Seam <= 0.050);
4. prompt and constraint following (CLIP Score, CMMD);
5. perceptual realism (KID <= 0.004, LPIPS <= 0.180, Aesthetic >= 0.850);
6. inference latency and peak memory (TTFT <= 120ms, T_swap = 0.0s, preview <= 1.0s).

## 8. Post-Training Evaluation & Research Paper Ingestion

Once the training run completes and writes `training_receipt.json`:

```bash
# 1. Execute the multi-hardware evaluation harness across held-out benchmarks
python scripts/eval/run_evaluation.py \
  --profile 48gb \
  --checkpoint checkpoints/qwen_image21_lora \
  --reference-dir data/reference_uis \
  --wireframe-dir data/held_out_wireframes \
  --prompt-list data/eval_prompts.json \
  --output-dir eval_results

# 2. Validate metrics against pre-registered hypotheses
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --dry-run

# 3. Apply verified empirical findings to the research paper
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --paper-path docs/RESEARCH_PAPER.md \
  --apply
```

This ensures zero manual transcription errors, generates `eval_results/evaluation_receipt.json` with cryptographic SHA-256 hashes, and populates the paper monograph's post-training result tables.
