# Data-Forge — Zero-Touch Data Pipeline (v16)

> Automated data pipeline for the **Krisna** project — a conversational, agentic
> UI-design system. This revision implements the PRD's **no-RLHF-loop** decision:
> every training-data source is either a real, human-labeled public dataset, or a
> model that ships **frozen** and needs no training data from this pipeline at all.
> There is no AI-judge/self-distillation step anywhere in this repository.
>
> Of the five components in the original PRD model stack, this pipeline trains
> **two**: the Sketch Tier (from scratch) and Z-Image-Turbo (fine-tune +
> Diffusion-DPO alignment). The Planner (Qwen3.5-9B), Qwen-Image-Edit-2511, and
> the Gemma-4 Critic all ship frozen — see "What This Pipeline Does NOT Train"
> below. See `../docs/data-forge/DATA_SOURCES.md` for the full source registry and
> `../docs/data-forge/DATA_COMPLETENESS.md` / `../docs/data-forge/ARCHITECTURE.md` for the reasoning trail
> behind this revision.

## What this pipeline actually produces

| Output | Feeds | Data source |
|---|---|---|
| `model_data/sketch_tier_maskgit/` | Sketch Tier, trained from scratch | RICO, CLAY, Enrico, WebUI (real UI screenshots) |
| `model_data/polish_zimage_turbo/` | Z-Image-Turbo, LoRA/QLoRA fine-tune | PD12M, CC12M + the UI sources above |
| `model_data/dpo_alignment/general/` | Z-Image-Turbo, Diffusion-DPO Stage 1 | Pick-a-Pic v2, HPDv2 (real human preference pairs) |
| `model_data/dpo_alignment/domain/` | Z-Image-Turbo, Diffusion-DPO Stage 2 | DesignSense-10k, DesignPref (real human-designer preference pairs — **not yet publicly released**, see below) |
| `model_data/planner_rag_corpus/` | Product-side RAG index (not a training set) | UICrit (real human critique text) |
| `model_data/eval_external/` | Held-out evaluation only | TASTE, PartiPrompts — structurally unreachable from any training path |

## What this pipeline does NOT train

- **Planner (Qwen3.5-9B)** ships frozen. It uses RAG over UICrit's real critique
  text (`model_data/planner_rag_corpus/`) plus constrained JSON decoding at
  inference time, in the product, not here.
- **Qwen-Image-Edit-2511** ships frozen. The "polish, quality path" uses
  zero-shot in-context-learning edit conditioning + SDEdit-style partial
  denoising at inference time — no paired edit-training data is generated or
  needed.
- **Gemma-4 Critic** ships frozen as an on-demand, product-side critique
  feature. It is never trained by this pipeline, and its output is never used
  as training data for anything else — there is no AI-judge-labeled data
  anywhere in this repository.

If you're looking for a `s07_5_edit_pairs`, `s01_6_planner_synthesis`, or
`s10_5_critic_preference` stage from an earlier revision: they were removed,
not renamed. The training tasks they fed no longer exist.

## Architecture

```
Orchestrator (Chunk-Based)
  Stage 0  Manifest Planning        -- Init DB, storage check, watcher report
  Stage 1  Fetch + License          -- Tier-1
  Stage 1.5  UICrit Join            -- --
  Stage 1.6  Preference Pairs       -- Tier-1 (independent stream, see below)
  Stage 2  Dedup (FAISS)            -- CLIP
  Stage 3  Quality Scoring          -- Tier-1
  Stage 3.5  PII Scrub              -- MediaPipe
  Stage 4  Safety                   -- Tier-1
  Stage 4.5  Escalation             -- Tier-2
  Stage 5  Recaption                -- Tier-1
  Stage 5-OCR  OCR Enrichment       -- DeepSeek-OCR
  Stage 5.5  PII Text Redact        -- Regex
  Stage 6  Structure Extract        -- Tier-1
  Stage 7  Routing & Shard          -- --
  Stage 8  Encoding (Z-Image + VQ)  -- VAEs/VQ
  Stage 8.5  DPO Encoding (disabled by default) -- VAE (see note below — Stage 1.6's output is actually consumed by training/'s sync_dpo_pairs.py, not this stage)
  Stage 9  Heldout Carve            -- --
  Stage 10  Audit Pass              -- Tier-1 + Tier-2
  Stage 11  Registry Watch          -- --
  Stage 12  Model Data Export       -- --
```

**Two independent streams, not one linear pipeline.** The main manifest stream
(Stage 0 -> 10) processes single UI/general images through dedup, quality,
safety, captioning, structure extraction, and encoding — this is what feeds
the Sketch Tier and Z-Image-Turbo's base fine-tune. Preference pairs (Pick-a-
Pic v2, HPDv2, DesignSense-10k, DesignPref) are a **separate stream**: Stage
1.6 dedups/blurs/safety-classifies them directly (they never enter the
SQLite manifest — a ranked pair isn't a single-image record). **Correction:**
this used to say Stage 8.5 "encodes them into DPO training latents
afterward" — that stage is disabled by default
(`configs/pipeline.yaml`: `enabled: false`) and has zero real consumers;
`training/data_forge_bridge/sync_dpo_pairs.py` reads Stage 1.6's raw pair
images directly via the shared BlobStore, and `train_dpo.py` live-encodes
them at train time instead. See `docs/data-forge/DATA_COMPLETENESS.md`
and `s08_5_dpo_encoding.py`'s own module docstring for the full history.
Stage 12 pulls from the main stream to build `model_data/`; the
preference-pair stream is consumed directly by the training bridge above,
not through Stage 12.

`05` and `05-OCR` are two distinct, separately-registered stages
(`s05_recaption` and `s05_ocr_enrichment`), each with its own dedicated module
under `data_forge/stages/` (`s05_recaption.py` and `s05_ocr_enrichment.py`).

### Python API Usage

Core abstractions can be imported directly from the top-level package:
```python
from data_forge import (
    Manifest,
    PipelineConfig,
    load_config,
    Orchestrator,
    register_all_stages,
)
```

## Requirements

- **OS**: Ubuntu 22.04 / 24.04 LTS (production), Windows (development)
- **CUDA**: 12.4 (strictly pinned)
- **Python**: 3.10 or 3.11
- **GPU**: 48GB VRAM (e.g., RTX 6000 Ada, A6000, L40S)
- **RAM**: 128GB system RAM recommended
- **Disk**: >=3TB at `DATA_ROOT`

## Setup & Verification

### 1. Environment Bootstrap
```powershell
cd data-forge
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1
conda activate krisna-forge
```
Installs Python 3.10, `torch+cu124`, `faiss-gpu`, `pandas`/`pyarrow` (needed for
`preference_pairs/`, `ui_critique/`, and DPO-latent metadata output), then
`pip install -e .[dev]`.

### 2. Pre-Flight Schema Verification
```powershell
python scripts/verify_schemas.py
```

### 3. Pipeline Dry-Run Validation
```powershell
$env:DATA_ROOT="D:\kf_data"
data-forge run --dry-run
```

## Usage

```powershell
$env:DATA_ROOT="D:\kf_data"
$env:HF_TOKEN="your_huggingface_token"

# Smoke test (100-record micro-batch)
data-forge run --chunk-size 100 --limit 100

# Full pipeline run
data-forge run

# Resume from checkpoint
data-forge run --resume

# Run specific stages
data-forge run --stages 0,1,2

# Run just the preference-pair post-processing (dedup/blur/safety)
data-forge run --stages 1.6

# Run just the DPO latent encoding (disabled by default — see the note
# above; only useful if you've re-enabled it for a specific reason)
data-forge run --stages 8.5

# Registry watcher (schedule via Task Scheduler) — also watches for
# DesignSense-10k/DesignPref public releases, see configs/datasets.yaml
data-forge registry check

data-forge manifest stats
data-forge manifest query --status excluded_pending_review
```

## Configuration

All configuration lives in `configs/`:
- `pipeline.yaml` — stage toggles, thresholds, paths, chunk sizes, storage estimates
- `models.yaml` — pinned model versions, quant settings, VRAM budgets. Only `z_image_vae` remains as an encoder — no `qwen_image_vae`, no `critic` model entry (both removed; those models are frozen, data-forge never loads them), and `maskgit_vq` was removed (sync audit item #1: it never had a working implementation, and its only caller was dead code — see `docs/review/01_sketch_tier.md`)
- `datasets.yaml` — 15 registered sources; see the table below

Paths in `pipeline.yaml` are **relative to `DATA_ROOT`**.

## Stage Reference

| Stage | Name | GPU Model | Purpose |
|-------|------|-----------|---------|
| 00 | Manifest Planning | -- | Init DB, storage check (image-record and preference-pair budgets tracked separately), read watcher report |
| 01 | Fetch & License | Tier-1 | Download + inline license verification. Five distinct fetch shapes: `url_list` (PD12M/CC12M), `hf_parquet_images` (RICO — images embedded in parquet, not loose files), `caption_join` (Screen2Words), `preference_pair`/`hpdv2_ranked_list` (DPO sources), `eval_reference` (TASTE/PartiPrompts) |
| 01.5 | UICrit Join | -- | Joins UICrit's real human critique/ratings onto already-ingested RICO records. Feeds the Planner's RAG corpus, not a training set |
| 01.6 | Preference Pairs | Tier-1 | Cross-source dedup + face-blur + **NSFW/harmful-content safety classification** on real DPO preference pairs. Refuses to run without a Tier-1 engine rather than skip the safety check |
| 02 | Dedup | CLIP | FAISS near-duplicate removal (main manifest stream only) |
| 03 | Quality | Tier-1 | Aesthetic/resolution scoring |
| 03.5 | PII Scrub | MediaPipe | Face blur, text redaction (shared `data_forge/utils/pii_faces.py`, also used by Stage 1.6) |
| 04 | Safety | Tier-1 | NSFW/harmful classification (main manifest stream) |
| 04.5 | Escalation | Tier-2 | Borderline second opinion |
| 05 | Recaption | Tier-1 | Dense structural captioning |
| 05-OCR | OCR Enrichment | DeepSeek-OCR | Text-in-image extraction |
| 05.5 | PII Text Redact | Regex | Redacts text found by the OCR pass |
| 06 | Structure | Tier-1 | UI layout JSON extraction |
| 07 | Routing | -- | Domain tagging + shard assignment (`ui_first` vs `general_design`) |
| 08 | Encoding | VAEs/VQ | Z-Image-Turbo latents + MaskGIT VQ tokens (`ui_first`-only) + control maps. **No Qwen-Image-Edit-2511 branch** — that model is frozen |
| 08.5 | DPO Encoding — **disabled by default** | VAE | Would encode Stage 1.6's preference pairs into Z-Image-Turbo's latent space; `train_dpo.py` live-encodes instead and never reads this stage's output — see the note above |
| 09 | Heldout Carve | -- | Stratified eval split; rejects encoding-incomplete records first |
| 10 | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge on 2-5% sample |
| 11 | Registry Watch | -- | Model/dataset release polling, including explicit watches for DesignSense-10k/DesignPref |
| 12 | Model Data Export | -- | Segments into `model_data/`: `sketch_tier_maskgit/`, `polish_zimage_turbo/`, `dpo_alignment/{general,domain}/`, `planner_rag_corpus/`, `eval_external/` |

## Source Registry Summary

See `../docs/data-forge/DATA_SOURCES.md` for full detail. Fifteen registered sources:

- **Foundation (real images, trained on directly):** PD12M, CC12M, RICO (core + semantic), CLAY, Enrico, WebUI, Screen2Words
- **DPO alignment (real human preference pairs):** Pick-a-Pic v2, HPDv2 (general); DesignSense-10k, DesignPref (UI/design-domain — **DesignPref is confirmed not yet publicly released**; DesignSense-10k has no confirmed public repo and a confirmed CC BY-NC-ND 4.0 license that would need separate legal sign-off even once released)
- **RAG corpus (not trained on):** UICrit
- **Evaluation only, structurally unreachable from training:** TASTE, PartiPrompts

## Known Issues, Fixed This Revision (v16)

- **RICO fetch produced zero records.** `rico_core`/`rico_semantic` were
  configured with `file_patterns: ["*.jpg","*.png","*.json"]` against two live
  HF repos that ship images embedded as bytes inside parquet files, not loose
  files — `allow_patterns` matched nothing, so every run silently fetched zero
  images from the single most foundational dataset in the corpus (CLAY,
  Enrico, and Screen2Words all join onto RICO records that would never have
  existed). New `download_mode: "hf_parquet_images"` decodes the embedded
  image column directly. See `../tests/data_forge/test_fetch_hf_parquet_images.py`.
- **RICO's two image-column names, fully confirmed, not auto-detected.**
  `rico_core` (creative-graphic-design/Rico) uses `screenshot`; `rico_semantic`
  (Voxel51/rico) uses `image` — confirmed directly against each live dataset
  card (rico_core's documented feature schema; rico_semantic's own Data
  Studio preview table header). The two repos are independently exported and
  do not share a column name despite covering the same underlying screens.
  Both are hardcoded in `datasets.yaml` now, not left to runtime
  auto-detection. `rico_semantic`'s license is also now confirmed CC BY 4.0
  directly from its dataset card (was a softer "believed" claim).
- **Preference-pair images were never screened for NSFW/harmful content.**
  Stage 1.6 previously ran with `engine=None` and did dedup/blur only. Now
  requires a Tier-1 engine and refuses to run without one; `unsafe`-tier pairs
  are dropped, `borderline` pairs kept but flagged. See
  `../tests/data_forge/test_preference_pairs_safety.py`.
- **HPDv2's real schema didn't match the generic preference-pair fetch path.**
  HPDv2 ships a variable-length ranked-list format (`human_preference:
  list[int]`, `file_path: list[str]`), not a fixed two-image table — the
  generic path would have found zero usable pairs. Dedicated
  `download_mode: "hpdv2_ranked_list"` adapter added.
- **The HPDv2 fix above initially caused a second-order bug**, twice: the new
  `"hpdv2_ranked_list"` mode wasn't recognized by Stage 1.6's source filter
  (would have skipped HPDv2's pairs entirely) or by the storage-projection
  methods in `config.py` (would have mis-projected HPDv2 as ordinary image
  records instead of preference pairs). Centralized into one
  `DatasetSpec.PREFERENCE_PAIR_DOWNLOAD_MODES` constant so the two can't drift
  apart again.
- **CLAY and PD12M's licenses were marked more conservatively than the facts
  support.** CLAY is CC BY 4.0 per its own paper's copyright line (was "unclear
  archived repo, likely excluded"). PD12M's CDLA-Permissive-2.0 and its
  `url`/`caption` column names are now confirmed against the live dataset
  card, not an unverified guess.
- **Storage-accounting double-counting.** `StorageManager.calculate_projected_
  size` used to sum every `per_record_estimates` key against one record count
  — adding preference-pair storage estimates would have multiplied a
  materially larger per-pair cost by the image-record count. Split into
  disjoint per-image / per-preference-pair estimate sets with separate counts.
- **`data_forge/utils/completeness.py` required an artifact (`qwen_image_latent`) that
  Stage 8 no longer produces for any record**, which would have flagged 100%
  of records as encoding-incomplete after the Qwen-Image-Edit-2511 branch was
  removed. Fixed to the current two-artifact (`z_image_latent`, `control_map`)
  + one UI-only artifact (`vq_tokens`) requirement.
- **`cli.py`'s `--stages` shortcut map** (`1.6`, `7.5`, `10.5`) still pointed
  at deleted stage names after the no-RLHF-loop restructuring — fixed.
- **`s12_model_data_export.py` linked processed artifacts but never the raw
  images either downstream consumer actually needs.** Found while wiring up
  krisna-orchestrator's sync bridge
  (`training/data_forge_bridge/`, that project's side): `sketch_tier_maskgit/`
  only linked `vq_tokens/` (Open-MAGVIT2 `.pt` files — that encoder's own
  loading code in `engine.py` raises `RuntimeError` on purpose, unverified
  working wrapper), and `polish_zimage_turbo/` only linked `latents/` (no
  consumer — the official `train_dreambooth_lora_z_image.py` script takes
  raw images and computes its own latents internally). Both exporters now
  also link `images/`. Caught a second, genuine bug in the same pass: with
  zero matching records for a domain, neither exporter created its own
  output directory before writing `captions.jsonl`, raising
  `FileNotFoundError` — `_export_planner_rag` already did this correctly,
  the other two didn't. See `../tests/data_forge/test_s12_images_export.py`.

## Still Open (documented, not hidden)

- **DesignSense-10k and DesignPref have no fetchable public data.** Re-checked
  directly against HuggingFace twice this revision, not just carried forward
  from an earlier pass. DesignPref is *confirmed* not yet released (two
  independent papers state this outright — its own describing paper and
  TASTE's related-work section). DesignSense-10k has no public repo found in
  either check, and its confirmed CC BY-NC-ND 4.0 license would need separate
  legal sign-off even once released (NoDerivs conflicts with this pipeline's
  recaption step; NonCommercial is a separate concern for any product track).
  Both `repo_id: null` in `datasets.yaml`, both tracked via
  `watcher_scan_targets` for a future release. **Practical effect:** Z-Image-
  Turbo's Stage-2 (UI-domain) DPO alignment has zero usable data until one of
  these is released — Stage-1 (general, Pick-a-Pic v2 + HPDv2) DPO runs as
  the only alignment signal in the meantime.
- **GameLabel-10K, a real, Apache-2.0, general-domain preference dataset,
  found during this revision's verification but not yet wired in** — its
  file is a single 2.26GB CSV, not parquet, so it doesn't fit either existing
  preference-pair fetch shape, and its exact column names weren't
  independently confirmed before this revision closed. See
  `../docs/data-forge/DATA_SOURCES.md`'s "Vetted but not yet integrated" section before
  adding it — inspect the real file header first rather than guessing a
  schema, same discipline that caught the PD12M/HPDv2 near-misses above.

**Prior revisions (v14-v15), still in effect:** OCR enrichment's missing
config block, the never-published Qwen-Image-2.0-VAE reference, a full-table
manifest scan repeated per dataset, an arbitrary Windows MAX_PATH threshold,
dead FAISS config, an unclosed `httpx.AsyncClient` in the registry watcher,
three UI datasets missing from the domain tagger, and two duplicated
`StageResult` classes.

## Testing

```bash
pytest tests/ -v -m "not integration"      # Unit tests, no GPU required
pytest tests/ -v -m integration            # Requires GPU + models downloaded
pytest tests/ --cov=data_forge --cov-report=html
```

100+ tests, all passing as of this revision (verified via `py_compile` across
every module, a full `load_config()` pass against all three YAML files, the
orchestrator's stage-registration/ordering validator, and a fresh zip
extraction re-run — not just `pytest`'s own report).
