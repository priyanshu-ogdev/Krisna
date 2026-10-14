# Flow-Matching Architecture Upgrade Plan

**Date:** 2026-10-15
**Evidence policy:** publisher proceedings, official model cards, and official
implementation documentation only. No arXiv-only source is used for the
decisions below.

## 1. What flow-matching model is actually in scope?

The active flow-matching model is **Qwen-Image-2.1**. It is the only model
with an intended trainable artifact:

```text
Qwen-Image-2.1 native pipeline
  frozen base transformer
  + supervised image-conditioned LoRA
  condition: deterministic Layout IR wireframe
  target: source screenshot
  text: controlled edit instruction/caption
```

The retired Z-Image/DPO implementation is not an alternative Qwen trainer. Its
velocity objective, latent contract, and model-specific call shape must not be
copied into Qwen training.

The Qwen model card documents a 7B visual-generation component, a native
`QwenImage21Pipeline`, image editing, image conditioning, transparent image
support, multiple references, and `enable_model_cpu_offload()`. The official
Diffusers training README documents the paired image-to-image trainer and the
important fact that the condition image is consumed both as vision tokens and
as clean latents before the noisy target sequence.

## 2. Research-grounded design decisions

### 2.1 Rectified-flow and DiT principles

Esser et al. in the peer-reviewed ICML 2024 paper *Scaling Rectified Flow
Transformers for High-Resolution Image Synthesis* show the value of rectified
flow paths, resolution-aware noise-scale design, and bidirectional text/image
token interaction in a transformer generator. This supports using the native
Qwen scheduler and pipeline rather than implementing a second ODE solver in
Krisna.

**Upgrade:** the repository no longer reports its compatibility Euler/RK2
helper as the real Qwen solver. Real receipts record the loaded pipeline's
actual scheduler class and timesteps.

### 2.2 Image-conditioned editing

InstructPix2Pix (CVPR 2023 Open Access) establishes instruction-conditioned
image editing without per-example optimization. ControlNet (ICCV 2023 Open
Access) establishes a complementary pattern: preserve a strong generator and
learn an explicit spatial conditioning path.

**Upgrade:** Krisna keeps the deterministic wireframe as the condition and
freezes the Qwen base. The first experiment trains only a LoRA. A future
ControlNet-like or mask-aware branch is a separate ablation, not an implicit
change to the primary trainer.

### 2.3 Inversion-free flow editing

FlowEdit (ICCV 2025 Open Access) demonstrates inversion-free, optimization-free
editing for pretrained flow models and reports transfer across Stable Diffusion
3 and FLUX. It is relevant to inference-time editing, but it does not prove
compatibility with Qwen-Image-2.1 or with the wireframe-to-UI task.

**Upgrade decision:** do not add FlowEdit to the default path yet. First measure
the native Qwen image-conditioned path. Then compare native editing against an
inversion-free ablation using identical prompts, seeds, condition images, and
lock masks.

## 3. Training architecture — upgraded contract

### 3.1 Data contract

The training bridge now materializes split-isolated, typed Parquet datasets:

```text
prepared/
  train/
    data.parquet              # datasets.Image image + condition_image
    metadata.jsonl
    target_*.png / condition_*.png
  validation/
    data.parquet
    metadata.jsonl
  test/
    data.parquet
    metadata.jsonl
  metadata.jsonl                # train-only compatibility index
  dataset_receipt.json
```

The receipt binds:

- source manifest SHA-256;
- train/validation/test counts;
- per-split metadata and Parquet hashes;
- per-pair condition/target hashes;
- typed image columns;
- source-group split policy;
- source provenance and revision.

The conservative fallback is dataset-level grouping when an upstream source
does not supply a reliable application/page group. A group cannot cross
train/validation/test. A missing train split, missing typed image bytes,
identity pair, path escape, hash mismatch, or incomplete receipt fails before
the trainer starts.

### 3.2 Native trainer boundary

The external Diffusers checkout is pinned to:

```text
efe46a5b177af67c64b3666da8ddddac7ee12163
```

The CPU-only preflight verifies:

1. the checkout exists and its Git HEAD is exact;
2. the native Qwen image-to-image trainer exists;
3. the trainer contains the native Qwen pipeline and paired columns;
4. the prepared dataset receipt is valid;
5. only supported CLI arguments are emitted;
6. credentials are redacted from logs;
7. no `accelerate` subprocess starts until all checks pass.

The preflight intentionally allows a dirty external checkout only when its
underlying commit is still pinned. The dirty state is reported in the run
receipt and must be reviewed before publication.

### 3.3 Objective and parameter policy

- Base Qwen weights remain frozen.
- Only declared LoRA modules are trainable.
- `condition_image`, `image`, and `caption` are explicit columns.
- `bf16` is required by the active configuration.
- Random horizontal flips are disabled because they alter UI text and layout
  semantics.
- Aspect-ratio buckets are explicit and divisible by the native trainer's
  geometry requirement.
- `val_fraction` and `val_every` are rejected; held-out data is materialized as
  a real validation/test split before launch.
- Parameter counts are not invented in manifests; they are filled only by a
  real trainer smoke test.

### 3.4 Required GPU smoke test

The first GPU run is not a quality run. It must prove the model contract:

```text
load native pipeline
  -> inspect base requires_grad flags
  -> inspect LoRA trainable parameters
  -> consume one condition/target pair
  -> perform one optimizer step
  -> confirm LoRA weights changed
  -> save adapter
  -> reload adapter
  -> generate with the same native pipeline and condition contract
```

The resulting `training_receipt.json` must include the exact model revision,
Diffusers commit, dataset hash/counts, trainable/frozen counts, target modules,
seed, precision, hardware, peak VRAM, and save/load result.

## 4. Inference architecture — upgraded contract

### 4.1 Native API authority

The installed `QwenImage21Pipeline.__call__` signature is inspected at load
time. Real calls pass only parameters supported by that exact pipeline:

```text
prompt
image
num_inference_steps
generator (when a seed is requested)
true_cfg_scale (only when supported)
```

Legacy `strength`, invented image-guidance schedules, and custom Euler/RK2
controls are not forwarded to the native Qwen pipeline. If strict mode receives
one of those controls, it fails rather than pretending the control was used.

### 4.2 Accurate receipts

Real inference receipts now distinguish:

- requested steps versus actual steps;
- the native scheduler class;
- live native timesteps, when exposed;
- the pipeline class and pinned model revision;
- the seed and generator device;
- native versus synthetic runtime mode;
- fallback reason, if a non-strict test path used synthetic output.

The compatibility `FlowODEScheduler` remains only for CPU/synthetic tests and
is never represented as the real Qwen solver.

### 4.3 Adapter and tier isolation

- Polish Default may load the explicitly configured adapter.
- Polish Quality rejects every adapter path, including explicit paths.
- Strict serving requires a manifest with exact base model, exact base revision,
  native pipeline, image conditioning, dataset hash, split counts, and LoRA
  target modules.
- Product fallback behavior remains available by default, but reproducible
  experiments can set `allow_tier_fallback=False` so an OOM or load failure
  cannot silently change the evaluated model.

### 4.4 Exact locks

The compositor now uses half-open rectangles and rejects malformed or ambiguous
normalized/absolute coordinates. Exact protected-pixel equality is measured
only when source and rendered dimensions match; resizing is not silently
accepted as proof of pixel identity. Seam quality is reported separately from
protected-region equality.

## 5. Experiment plan

### Phase A — contract gates

1. Run Data-Forge strict local preprocessing on a CUDA host.
2. Materialize a non-empty, source-disjoint Qwen pair corpus.
3. Run the CPU preflight and review the redacted command/receipt.
4. Run the one-batch native Qwen train/save/load/inference smoke test.

### Phase B — supervised adapter

Use a stratified 1–5% smoke corpus first. Sweep only:

| Variable | Values |
|---|---|
| LoRA rank | 16, 32, 64 |
| Learning rate | 5e-5, 1e-4 |
| Effective batch | 4, 8, 16 |
| Target passes | 1, 2, 3 |
| Seed | at least 3 final candidates |

Select by exact OCR/CER, layout adherence, protected-pixel equality, and
handoff consistency. Aesthetic score is secondary.

### Phase C — controlled flow/editing ablations

1. Native Qwen image-conditioned editing.
2. Native Qwen plus explicit supported mask/control input, if available in the
   pinned pipeline.
3. Inversion-free FlowEdit-style inference, only if implemented against Qwen
   and measured with the same receipt schema.
4. ControlNet-like spatial adapter only as a separately trained experiment.

## 6. Publisher and official sources

- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image
  Synthesis*, ICML 2024, PMLR 235:
  <https://proceedings.mlr.press/v235/esser24a.html>
- Brooks et al., *InstructPix2Pix: Learning To Follow Image Editing
  Instructions*, CVPR 2023 Open Access:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html>
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*,
  ICCV 2023 Open Access:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html>
- Kulikov et al., *FlowEdit: Inversion-Free Text-Based Editing Using
  Pre-Trained Flow Models*, ICCV 2025 Open Access:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html>
- Qwen-Image-2.1 official model card:
  <https://huggingface.co/Qwen/Qwen-Image-2.1>
- Official pinned Diffusers Qwen-Image-2.1 training README:
  <https://raw.githubusercontent.com/huggingface/diffusers/efe46a5b177af67c64b3666da8ddddac7ee12163/examples/dreambooth/README_qwenimage21.md>
- Official pinned Diffusers Qwen-Image-2.1 pipeline source:
  <https://github.com/huggingface/diffusers/blob/efe46a5b177af67c64b3666da8ddddac7ee12163/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py>

IEEE Xplore, SpringerLink, and ScienceDirect search endpoints were attempted,
but authenticated access was not completed in the browser. No credentials
provided in chat were entered or stored, and no paywalled claim is used as
evidence in this plan.
