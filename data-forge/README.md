# Data-Forge — Zero-Touch Data Pipeline (v15)

> Automated data pipeline for the **Krisna** project. Implements the v15 specification:
> every verification, judgment, and spot-check step that was previously manual is now
> handled by deterministic logic or a resident reasoning/VLM model via vLLM, plus an
> on-demand Critic Tier (Gemma 4 31B) for alignment-training signal generation, and —
> new in v15 — dedicated data production for every one of the PRD's five trainable
> components, not just the image-preprocessing majority. See
> `docs/DATA_COMPLETENESS.md` for the full model-to-data trace that drove this revision.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Orchestrator (Chunk-Based)                             │
│  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌─────────┐┌──────┐     │
│  │Stage 0 ││Stage 1 ││  1.5   ││  1.6   ││Stage 2 ││Stage 3  ││ 3.5  │     │
│  │Manifest││Fetch+  ││UICrit  ││Planner ││Dedup   ││Quality  ││ PII  │     │
│  │Planning││License ││Join    ││Synth   ││(FAISS) ││Scoring  ││Scrub │     │
│  └────────┘└────────┘└────────┘└────────┘└────────┘└─────────┘└──────┘     │
│       ↓                                                                      │
│  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐              │
│  │Stage 4 ││  4.5   ││Stage 5 ││05-OCR  ││  5.5   ││Stage 6 │              │
│  │Safety  ││Escalate││Recap   ││Enrich  ││PII Text││Struct  │              │
│  │Tier-1  ││Tier-2  ││Tier-1  ││DeepSeek││Redact  ││Extract │              │
│  └────────┘└────────┘└────────┘└────────┘└────────┘└────────┘              │
│       ↓                                                                      │
│  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐              │
│  │Stage 7 ││  7.5   ││Stage 8 ││Stage 9 ││Stage 10││  10.5  │              │
│  │Routing ││Edit    ││Tri-Path││Heldout ││Audit   ││Critic  │              │
│  │& Shard ││Pairs   ││Encode  ││Carve   ││Pass    ││Prefer. │              │
│  └────────┘└────────┘└────────┘└────────┘└────────┘└────────┘              │
│       ↓                                                                      │
│  ┌────────┐┌────────┐                                                       │
│  │Stage 11││Stage 12│                                                       │
│  │Registry││Model   │                                                       │
│  │Watcher ││Data    │                                                       │
│  │        ││Export  │                                                       │
│  └────────┘└────────┘                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

`05` and `05-OCR` are two distinct, separately-registered stages (`s05_recaption` and `s05_ocr_enrichment`) — a previously-fixed missing config block for the latter had silently disabled OCR extraction, and with it, all downstream text-PII redaction. New in v15: `1.5`/`1.6`/`7.5`/`12` close the gap between "the model stack is finalized" and "the pipeline can actually train all five of them" — see `docs/DATA_COMPLETENESS.md`.

## Requirements

- **OS**: Ubuntu 22.04 / 24.04 LTS (production), Windows (development)
- **CUDA**: 12.4 (strictly pinned)
- **Python**: 3.10 or 3.11
- **GPU**: 48GB VRAM (e.g., RTX 6000 Ada, A6000, L40S)
- **RAM**: 128GB system RAM recommended
- **Disk**: ≥3TB at `DATA_ROOT`

## Setup & Verification

To properly run the data-forge on Windows, you must use the provided bootstrap script. The script resolves pip compatibility issues by using Conda to install `faiss-gpu` and the correct CUDA 12.4 PyTorch wheels.

### 1. Environment Bootstrap
```powershell
# Navigate to the project root
cd data-forge

# Run the Windows environment setup script (requires Conda installed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1

# The script creates the environment. You must manually activate it before proceeding:
conda activate krisna-forge
```
*Linkage*: The script handles Python 3.10 initialization, `torch+cu124`, `faiss-gpu`, and finally executes `pip install -e .[dev]` to link the `data-forge` CLI commands globally to this environment. This revision adds `pandas`/`pyarrow` to the dependency set (previously missing despite `/ui_critique/` and `/preference_pairs/` being documented Parquet outputs with no code to actually write them).

### 2. Pre-Flight Schema Verification
Before loading any heavy GPU models, validate that your YAML configurations are well-formed and internally consistent.
```powershell
python scripts/verify_schemas.py
```

### 3. Pipeline Dry-Run Validation
Verify the SQLite manifest connection (WAL-mode locking) and storage quota limits without spinning up the inference engine.
```powershell
$env:DATA_ROOT="D:\kf_data"
data-forge run --dry-run
```

## Usage

Once setup and verification are complete, you can execute the pipeline natively.

```powershell
# Set required credentials
$env:DATA_ROOT="D:\kf_data"
$env:HF_TOKEN="your_huggingface_token"

# Execute the Smoke Test (100-record micro-batch)
data-forge run --chunk-size 100 --limit 100

# Full pipeline run
data-forge run

# Resume from checkpoint
data-forge run --resume

# Run specific stages (e.g., fetch, dedup, quality)
data-forge run --stages 0,1,2

# Run the new Critic Tier stage on its own
data-forge run --stages 10.5

# Registry watcher (schedule this via Windows Task Scheduler)
data-forge registry check

# Inspect manifest
data-forge manifest stats
data-forge manifest query --status excluded_pending_review
```

## Configuration

All configuration lives in `configs/`:
- `pipeline.yaml` — stage toggles, thresholds, paths, chunk sizes
- `models.yaml` — pinned model versions, quant settings, VRAM budgets (now includes the `critic` model entry for Gemma 4 31B Dense)
- `datasets.yaml` — source dataset URLs, license status, categories (PD12M/CC12M now specify `download_mode: "url_list"` with sampling caps — see `docs/DATA_SOURCES.md`)

Paths in `pipeline.yaml` are **relative to `DATA_ROOT`**.

## Stage Reference

| Stage | Name | GPU Model | Purpose |
|-------|------|-----------|---------|
| 00 | Manifest Planning | — | Init DB, storage check, read watcher report |
| 01 | Fetch & License | Tier-1 | Download (incl. URL-list path for metadata-only sources) + inline license verification |
| 01.5 | UICrit Join | — | Joins UICrit's real human critique/ratings against already-ingested RICO records |
| 01.6 | Planner Synthesis | Tier-1 | Generates Planner (Qwen3.5-9B) conversational SFT data, seeded from real UICrit critique text |
| 02 | Dedup | CLIP | FAISS near-duplicate removal |
| 03 | Quality | Tier-1 | Aesthetic/resolution scoring |
| 03.5 | PII Scrub | MediaPipe | Face blur, text redaction |
| 04 | Safety | Tier-1 | NSFW/harmful classification |
| 04.5 | Escalation | Tier-2 | Borderline second opinion |
| 05 | Recaption | Tier-1 | Dense structural captioning (uses `source_caption` as a prior hint when the source dataset provides one) |
| 05-OCR | OCR Enrichment | DeepSeek-OCR | Text-in-image extraction (distinct registered stage) |
| 05.5 | PII Text Redact | Regex | Redacts text found by the OCR pass |
| 06 | Structure | Tier-1 | UI layout JSON extraction |
| 07 | Routing | — | Domain tagging + shard assignment |
| 07.5 | Edit Pairs | — | Synthetic (rough sketch, target) pairs for Qwen-Image-Edit-2511's actual edit-conditioned task |
| 08 | Tri-Path Encode | VAEs/VQ | Z-Image, Qwen-Image-Edit-2511, MaskGIT tokens (VQ tokens now `ui_first`-only) |
| 09 | Heldout Carve | — | Stratified eval set split (now rejects encoding-incomplete records first) |
| 10 | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge on 2-5% sample |
| 10.5 | Critic Preference | Gemma 4 31B | Bulk critique generation → `ui_critique/*.parquet` |
| 11 | Registry Watch | — | Model/dataset release polling |
| 12 | Model Data Export | — | Segments the manifest into `model_data/`, one clean folder per PRD-trainable component |

## Known Fixes in This Revision (v15)

**Data completeness — see `docs/DATA_COMPLETENESS.md` for the full trace:**
- **UICrit's real human annotations were completely discarded on ingestion.** `_fetch_github` unconditionally globbed for image extensions regardless of `fetch_config`, so UICrit's `.json`/`.csv` critique/rating files were cloned and then silently never parsed. This was the direct cause of the Critic Tier having no real calibration data — only Gemma 4's own self-generated critiques (self-distillation, not calibration). Fixed via the new `annotation_only` dataset flag and `s01_5_uicrit_join`.
- **Screen2Words silently ingested zero records** — same failure class as PD12M/CC12M before the prior fix, just not caught in that pass. Fixed via a new `caption_join` download mode with defensive shape auto-detection.
- **The Planner had no training data path at all.** Fixed via `s01_6_planner_synthesis`, seeded from UICrit's real critique text (once §1 was fixed) plus Glaive/xLAM format-teacher datasets for output shape.
- **Qwen-Image-Edit-2511 only ever saw single finished images**, not the (rough sketch, instruction, polished target) triples its actual edit-conditioned task needs. Fixed via `s07_5_edit_pairs` (synthetic UI-domain pairs) plus MagicBrush/InstructPix2Pix ingestion (general-domain task-shape grounding).
- **Stage 8's VQ-token branch ran on every domain**, wasting compute encoding tokens for general-domain images the UI-only sketch tier will never use. Domain-gated to `ui_first` only.
- **`status == "encoded"` was treated as "fully encoded"** even when a record was missing one or more artifacts its domain actually requires. `s09_heldout` now checks real completeness via the new domain-aware `utils/completeness.py` predicate before admitting records to `training_pool`/`heldout`.
- **`source_caption` was captured by the fetcher but silently dropped** — `bulk_create_records` only ever read a fixed set of dict keys, so a field extracted from PD12M/CC12M/Screen2Words never actually reached the database, and even if it had, nothing downstream used it. Fixed at both ends: stored properly, and now used as a prompt hint in `s05_recaption`.
- **No test caught any of the three silent zero-record ingestion bugs.** Added `tests/test_dataset_ingestion_completeness.py` and `tests/test_completeness_and_captions.py`.

**Prior revision (v14) fixes, still in effect:** OCR enrichment's missing config block (and the PII redaction it silently disabled), the never-published Qwen-Image-2.0-VAE reference, a full-table manifest scan repeated per dataset, an arbitrary Windows MAX_PATH threshold, dead FAISS config, an unclosed `httpx.AsyncClient` in the registry watcher, three UI datasets missing from the domain tagger, and two duplicated `StageResult` classes. One attempted "fix" to `ShardRouter`'s ratio enforcement was found to be wrong and reverted — see that file's docstring.

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/ -v -m "not integration"

# Integration tests (requires GPU + models downloaded)
pytest tests/ -v -m integration

# Coverage
pytest tests/ --cov=data_forge --cov-report=html
```

> Note: this revision's fixes were verified with targeted functional tests against a live SQLite manifest, real config loading, and isolated logic checks — including manually re-running every new test module's assertions directly (not just via `pytest`) — since this development environment has no outbound network access to install `pydantic`/`structlog`/`httpx`/`pyarrow`/`pytest`/etc. Run the real suite in a properly-connected environment before deploying.
