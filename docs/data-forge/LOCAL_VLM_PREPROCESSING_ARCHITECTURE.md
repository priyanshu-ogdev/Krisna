# Strict Local VLM Preprocessing Architecture

## Decision

The final model-data path uses a real local vision-language model rather than
the previous 9B bulk default or source-metadata shortcuts:

- **Model:** `Qwen/Qwen3.8-27B`
- **Revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- **Serving:** local vLLM OpenAI-compatible server
- **Quantization:** FP8
- **Target hardware:** one NVIDIA GPU with 48 GiB VRAM and a host with 128 GiB RAM
- **Primary role:** Tier-1 bulk preprocessing, OCR, structure extraction, quality,
  safety, captions, and audit inputs
- **Fallback policy:** none

The serving environment must be a CUDA-enabled Linux installation or WSL2
environment with the NVIDIA driver exposed. The current Windows/CPU-only
OmniRush runtime is intentionally rejected by the strict preflight; it is not
used as a hidden substitute.

The model card identifies Qwen3.8-27B as a native vision-language model and
documents vLLM-compatible serving. The exact revision is pinned in
`configs/models_local.yaml`; the mutable `main` branch is not accepted by the
strict CLI path.

Reference: <https://huggingface.co/Qwen/Qwen3.8-27B>

## Audit findings that prompted the change

The previous local configuration had several gaps:

1. `models_local.yaml` defined a 9B model as Tier-1 even though a 27B model was
   available for escalation.
2. The normal CLI always loaded `pipeline.yaml` and `models.yaml`; it did not
   select the local profile, so the local configuration was not an executable
   path.
3. The pipeline could silently return from VLM stages when no engine was
   available.
4. Structure extraction could reuse source structure or emit an empty
   `freeform` structure for general visual data.
5. OCR could emit an empty result for excluded/general visual records.
6. `max_num_seqs: 256` was too aggressive for a 27B multimodal model on a
   single 48 GiB GPU.
7. The image cache had no profile-level resource contract, so host RAM usage
   was implicit rather than bounded by the local hardware plan.

The strict profile addresses these gaps without changing the default profile
or silently changing existing user data.

## Production pipeline

The final training-data path is the governed pipeline, not a synthetic data
shortcut:

```text
source artifacts
  -> s00 manifest planning
  -> s01 fetch/provenance
  -> s02 exact + CLIP deduplication
  -> s03 Qwen3.8 quality scoring
  -> s03.5 deterministic image PII scrubbing
  -> s04 Qwen3.8 safety classification
  -> s04.5 same-model escalation for borderline records
  -> s05 Qwen3.8 dense caption
  -> s05 OCR Qwen3.8 extraction
  -> s06 Qwen3.8 canonical structure extraction
  -> s05.5 deterministic text PII redaction
  -> s07 domain routing
  -> s08 Z-Image VAE encoding
  -> s09 source-disjoint held-out split
  -> s10 Qwen3.8 audit pass
  -> s11 registry receipt
  -> s12 governed model-data export
```

The VLM-required stages all share the local Tier-1 model contract. A retry is
allowed for a transient request or schema failure, but a retry never changes
model, device, source structure, or output semantics.

The strict path rejects rather than substitutes when:

- CUDA or vLLM is unavailable;
- the live server is not serving the configured Tier-1 model;
- the model is text-only or lacks the `vision` capability;
- the model revision is not an immutable SHA;
- a caption, OCR, quality, safety, or structure response is missing;
- the output fails Pydantic/JSON-schema validation;
- the source structure is incomplete or non-canonical;
- a general-design record would otherwise receive a dummy empty structure;
- the training target or condition artifact is missing downstream.

No CPU inference, text-only fallback, weaker-model fallback, source-structure
substitution, empty/freeform output, synthetic label, or silent VLM-stage skip
is allowed in this profile.

## Hardware allocation

### 48 GiB VRAM

- Qwen3.8-27B FP8 is the resident multimodal VLM.
- `gpu_memory_utilization: 0.92` leaves an explicit runtime margin.
- `max_num_seqs: 48` and stage concurrency `48` are the initial safe settings;
  they should be benchmarked on the actual GPU before increasing them.
- The pipeline does not keep CLIP, VAE, and the VLM resident simultaneously.
  Model sessions remain lifecycle-scoped to avoid fragmentation and OOM.

### 128 GiB host RAM

- The local profile uses bounded source batches and bounded VLM concurrency.
- Image payloads are cached by path, modification metadata, and resize budget;
  the configured maximum image dimension is 2048 pixels.
- Host RAM is used for parquet batches, image encoding, manifest transactions,
  and bounded request queues, not as an unbounded model fallback.
- Swap space is a vLLM safety reserve only; it is not a permission to run the
  model on CPU.

## Commands

### Model-free source normalization

This remains useful for creating provenance-backed `fetched` records:

```powershell
python -m data_forge.cli preprocess `
  --dataset figma2code `
  --source-path D:\datasets\Figma2Code `
  --limit 2500
```

### Strict VLM bundle smoke/audit

This command requires the pinned local model, CUDA, and vLLM before it starts:

```powershell
python -m data_forge.cli preprocess-vlm `
  --dataset figma2code `
  --status fetched `
  --limit 100
```

It validates caption, structure, OCR, quality, and safety outputs, atomically
updates the manifest, and writes a receipt under `audit_reports/`. Failed
records are explicitly excluded. This is useful for a bounded VLM contract
smoke test; the full governed training path remains the command below.

### Final governed local pipeline

```powershell
python -m data_forge.cli run `
  --profile local-vlm `
  --resume `
  --stages 0,1,2,3,3.5,4,4.5,5,5-ocr,6,5.5,7,8,9,10,11,12
```

For the first hardware validation, use `--limit 16` and inspect the VLM log,
manifest receipt, GPU memory, excluded-record reasons, and stage outputs before
removing the cap.

## Validation gates

Before training is allowed, the run must have all of the following:

1. The strict local profile loads and reports the pinned Qwen model revision.
2. CUDA reports at least the configured VRAM budget.
3. vLLM becomes healthy without a subprocess fallback.
4. Every VLM-required record has schema-valid outputs.
5. Failed records have explicit exclusion reasons and are not exported.
6. PII scrub and text-redaction artifacts exist before training export.
7. Held-out source identities are disjoint from training identities.
8. `s10_audit` meets its configured pass-rate threshold.
9. `s12_model_data_export` produces a receipt with no policy violation.
10. The deterministic condition renderer has produced real condition images;
    JSON control maps or Canny edges alone are not accepted as Qwen pairs.

The current workspace can validate configuration, strict guards, mock VLM
contracts, and CLI wiring. A real GPU smoke run is still required to measure
actual model load time, peak VRAM, tokens/sec, and the safe concurrency ceiling.
