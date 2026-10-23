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

```bash
./scripts/training/train_polish_default_lora.sh \
  training/configs/polish_qwen_image21_lora.yaml
```

The configuration selects:

```text
base model: Qwen/Qwen-Image-2.1 (revision d26bb61231c349cf6b7896fa83353113880e1ba3)
conditioning column: condition_image
target column: image
caption column: caption
resolution: 1024
adaptation: dora (use_dora: true)
DoRA rank: 64
DoRA alpha: 32 (alpha = rank/2)
aspect_ratio_buckets: 1024,1024;768,1344;1344,768;832,1248;1248,832;896,1152;1152,896
dual operational profiles:
  - fast smoke/test profile: 5,000 steps (default, ~2.7h on A6000 Ada)
  - production SFT profile: 27,500 steps (2.01 epochs on 218,791 pairs, ~15.2h on A6000 Ada)
```

The rank, alpha, step count, and scheduler are formally calibrated. Scale steps from the realized manifest size (218,791 pairs) and effective batch (16): $\lceil 2.0 \times 218,791 / 16 \rceil = 27,500$ steps.


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

1. exact text/OCR error;
2. layout/node and box adherence;
3. protected-pixel equality for locked regions;
4. prompt and constraint following;
5. handoff consistency;
6. safety;
7. human preference and perceptual metrics;
8. latency and peak memory.

Preference alignment is a separate experiment. It must not be used to hide an
unvalidated supervised trainer or a missing condition-image path.
