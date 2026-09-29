# Data-Forge Architecture Specification (v16 — No-RLHF-Loop)

## Overview

The Data-Forge is a custom, 20-stage Python orchestrator designed to process real, published data — images and human-labeled preference pairs — into training-ready latents on a single workstation (specifically a 48GB VRAM GPU like an RTX A6000). It replaces manual curation with **VLM-as-Judge** auditing (for content gates, not preference labeling) and deterministic logic, enforcing a strict zero-touch automation philosophy.

**v16's defining change:** three of the PRD's five model components — the Planner, Qwen-Image-Edit-2511, and the Gemma-4 Critic — now ship **frozen**, requiring no fine-tuning and no training data at all. Data-forge only trains the sketch tier (from scratch) and Z-Image-Turbo (LoRA/QLoRA + Diffusion-DPO). Every step that previously depended on synthetically-generated or AI-judge-labeled data has been removed and replaced with either a real, already-published, human-labeled dataset or a training-free inference-time technique — see `DATA_COMPLETENESS.md` for the full model-to-data trace.

## The Orchestrator (Chunk-Based DAG)

Preprocessing large image corpora through multiple VLMs (Qwen, VAEs, OCR, Safety) on a single GPU normally suffers from severe PCIe swapping bottlenecks. The Data-Forge solves this with a **Chunk-Based Directed Acyclic Graph (DAG)**:

1. **Chunking**: The orchestrator splits the dataset into chunks (default 10,000 records).
2. **Model Pinning**: A required model (e.g., Tier-1 VLM) is loaded into VRAM.
3. **Execution**: The orchestrator runs all records in the chunk through every stage that requires that specific model.
4. **Teardown**: The model is gracefully unloaded, CUDA cache is cleared, and the next model is loaded.

`EXECUTION_ORDER` in `orchestrator.py` is the single source of truth for stage sequencing, cross-checked at startup by `validate_stage_ordering()` against every registered stage's declared `requires` — a stage registered but missing from `EXECUTION_ORDER`, or declaring a dependency that `EXECUTION_ORDER` doesn't actually satisfy, fails loudly at startup rather than silently misbehaving mid-run.

## Pipeline Stages

| Stage | Name | GPU Model | Purpose |
|---|---|---|---|
| **00** | Manifest Planning | — | Init SQLite DB, storage pre-flight check (image and preference-pair estimates computed and checked separately), read registry watcher report. |
| **01** | Fetch & License | Tier-1 (inline) | Downloads via one of six fetch shapes (see below), runs the License Verification Agent to extract terms and triage. |
| **01.5** | UICrit Join | — | Parses UICrit's real human critique/ratings and joins them onto already-ingested RICO records by filename-stem match. Feeds the Planner's RAG corpus (`s12`'s `planner_rag_corpus/` export) — not a fine-tune. |
| **01.6** | Preference Pairs | Tier-1 | Cross-source dedup, face-blur, and NSFW/safety classification on real human DPO preference pairs (Pick-a-Pic v2, HPDv2, DesignSense-10k, DesignPref) before they're eligible for encoding. |
| **02** | Dedup | CLIP / FAISS | Exact-hash + semantic near-duplicate removal on the main image manifest. |
| **03** | Quality | Tier-1 | Aesthetic and resolution scoring. |
| **03.5** | PII Scrub | MediaPipe/Regex | Face blurring and sensitive-text redaction. |
| **04** | Safety | Tier-1 | NSFW/harmful content classification (main image manifest). |
| **04.5** | Escalation | Tier-2 | Second opinion on borderline safety/license records. Routes to `excluded_pending_review`. |
| **05** | Recaption | Tier-1 | Dense structural captioning, using `source_caption` (the source dataset's own label, when one exists) as a prior hint rather than captioning from nothing. |
| **05-OCR** | OCR Enrichment | DeepSeek-OCR | Text-in-image extraction — a distinct, separately-registered stage. |
| **05.5** | PII Text Redact | Regex | Redacts sensitive text found by the OCR pass. |
| **06** | Structure | Tier-1 | UI component tree and bounding box JSON extraction. |
| **07** | Routing | — | Domain tagging, `ui_first_ratio` enforcement, shard assignment. |
| **08** | Tri-Path Encoding | VAEs / VQ | Encodes Z-Image-Turbo latents and MaskGIT VQ tokens (`ui_first`-only). **No Qwen-Image-Edit-2511 branch** — that model is frozen and never trained, so data-forge never encodes latents for it. |
| **08.5** | DPO Encoding — **disabled by default** | Z-Image VAE | Was meant to encode deduped, safety-checked preference pairs into Z-Image-Turbo's latent space ahead of time. Confirmed to have zero real consumers — `train_dpo.py` resolves pairs through the shared BlobStore and live-encodes them itself at train time instead (see `s08_5_dpo_encoding.py`'s own module docstring and `docs/review/15_data_forge_finalization.md`). `configs/pipeline.yaml` sets `enabled: false`; left in the codebase as an option (a precomputed-latent fast path, if DPO training ever becomes GPU-time-constrained) rather than deleted. |
| **09** | Heldout Carve | — | Stratified evaluation set carve-out. Rejects encoding-incomplete records (via the domain-aware `../../data-forge/data_forge/utils/completeness.py` predicate) before admitting anything to `training_pool`/`heldout`. |
| **10** | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge rubric evaluation (content/structure quality spot-check, not preference labeling) on 2-5% of `training_pool`. |
| **11** | Registry Watch | — | External cron; polls HF/GitHub for model/dataset updates, writes report. |
| **12** | Model Data Export | — | Segments the manifest into `model_data/`, one clean folder per component data-forge actually produces data for. |

### The six fetch shapes (Stage 1)

Not every source is a folder of standalone image files, and treating them as if they were is the single most common failure mode this pipeline has hit and fixed:

| `download_mode` | Used by | Why |
|---|---|---|
| (default — bundled files) | CLAY, Enrico, WebUI | Repo genuinely ships loose image files. |
| `hf_parquet_images` | RICO core, RICO semantic | Ships images embedded as bytes inside parquet (the modern `datasets`-library convention) — a plain file-extension scan finds nothing. Both repos' image column names are confirmed and hardcoded (`"screenshot"` for `creative-graphic-design/Rico`, `"image"` for `Voxel51/rico`) rather than auto-detected, since the two independently-exported repos of the same underlying screens don't share a column name. |
| `url_list` | PD12M, CC12M | Metadata-only repos: a parquet/TSV with an image-URL column, not bundled files. |
| `caption_join` | Screen2Words | Captions *for* RICO's screens, not new images of its own — joined onto already-ingested RICO records by ID. |
| `preference_pair` | Pick-a-Pic v2, DesignSense-10k, DesignPref | Fixed two-image-per-row `(prompt, image_a, image_b, label)` parquet shape. |
| `hpdv2_ranked_list` | HPDv2 | This dataset's real format is a variable-length ranked list per prompt, not a fixed two-image table — needs its own adapter, confirmed against the live dataset card. |
| `eval_reference` | TASTE, PartiPrompts | Structurally routed to `heldout/external_eval/` only — physically never reaches `training_pool` or `preference_pairs/`, by construction, not by a filter that could be accidentally bypassed. |

## Latent & Reference Data Storage Strategy

Approved records are converted into different representations depending on what actually consumes them — there are three disjoint output families, not one blanket scheme:

**Image-manifest artifacts (Stage 8):**
1. `latents_zimage/` (fp16 `.safetensors`) — continuous latents for Z-Image-Turbo.
2. `vq_tokens_sketch/` (int16 `.pt`) — discrete codebook indices for the sketch tier, `ui_first` domain only.
3. `control_tokens/` (`.json`) — structural bounding boxes and Canny edges.

**DPO alignment artifacts (Stage 8.5 — disabled by default, see the stage table above):**
4. `dpo_latents/{pickapic_v2,hpdv2,designsense_10k,designpref}/` — would hold both images of each surviving pair, encoded through Z-Image-Turbo's VAE, with prompt/preferred/origin metadata alongside, if this stage were enabled. In the actual training path, `train_dpo.py` never reads this directory — it live-encodes from `s01_6_preference_pairs`'s raw pair images instead.

**Reference/frozen-model support (no training, just retrieval or eval material):**
5. `planner_rag_corpus/` (`.jsonl`, via Stage 12) — UICrit's real critique text, retrieved at Planner inference time, never fine-tuned on.
6. `eval_external/{taste,partiprompts}/` (via Stage 12, linked read-only from `heldout/external_eval/`) — held-out evaluation material, structurally isolated from every training path.

*Storage checks are strictly enforced by `StorageManager` to preempt `MAX_PATH` errors on Windows and `ENOSPC` crashes — `calculate_projected_size()` computes image-record and preference-pair storage against separate, disjoint estimate-key sets and separate counts (`DatasetSpec.storage_relevant_record_count()`, which respects each dataset's `sample_size` cap rather than its raw `expected_record_count`), so the projection reflects what will actually be downloaded, not the full theoretical corpus size.*

## Sub-System Architecture

### 1. State Management (SQLite Manifest)
All state is tracked in a local SQLite database (`manifest.db`). The manifest operates in `WAL` mode with `isolation_level="IMMEDIATE"` and a 5000ms `busy_timeout` to prevent locking issues on Windows filesystems. Every record transition is logged in the `stage_history` table for a complete audit trail. A lightweight, idempotent column-migration step runs on every open, so a `manifest.db` from an older revision picks up newly-added columns (`pair_id`, `pair_role`, `is_eval_only`, `source_caption`, etc.) automatically rather than raising `no such column` the first time something writes to a field added after the database was first created.

Preference pairs are **not** manifest records in the same table as images — they're tracked as JSON sidecar files under `preference_pairs/{source}/` (written by Stage 1's dedicated fetch adapters, processed by Stage 1.6, then read directly by `training/data_forge_bridge/sync_dpo_pairs.py` — Stage 8.5 is disabled by default and isn't in this path, see the stage table above), since a pair's shape (two images, one label, one shared prompt) doesn't fit the single-image-per-row manifest schema cleanly. This is a deliberate architectural split, not an oversight — see `s01_6_preference_pairs.py`'s module docstring for the reasoning.

### 2. Inference Engine Lifecycle
The inference engine (`engine.py`) handles the lifecycle of models.
- **vLLM subprocess**: Tier-1 and Tier-2 are spawned as HTTP servers via vLLM. There is no Critic model session anymore — Gemma-4 31B is a frozen, on-demand, product-side feature that data-forge never loads. On teardown, `psutil` recursively kills all background CUDA worker threads before killing the parent process, ensuring zero VRAM leakage.
- **Transformers/PyTorch**: The CLIP model and Z-Image-Turbo's VAE (used by Stage 8; Stage 8.5 also uses it when explicitly re-enabled, but doesn't by default — see the stage table above) are loaded natively via PyTorch and cleared using `gc.collect()` and `torch.cuda.empty_cache()`. There is no `qwen_image_vae` encoder anymore — removed along with Qwen-Image-Edit-2511's training path. Each remaining encoder loads inside its own try/except, so one bad model reference degrades that one encoder's output rather than crashing the whole Stage 8 run.

### 3. VLM-as-Judge, Content Gates, and What They Are NOT
Automated VLM judgments handle content-quality and safety gating throughout: Stage 3 (aesthetic), Stage 4 (safety), Stage 10 (audit spot-check), and now Stage 1.6 (safety classification on preference-pair images). The pipeline uses an **Ensemble Disagreement** strategy for the audit pass — if Stage 10 disagrees with an upstream filter, the record is flagged and escalated rather than silently overridden. Ambiguous cases route to `excluded_pending_review` instead of halting the pipeline.

**None of this is preference *labeling*.** The old `s10_5_critic_preference.py` used Gemma-4 to generate its own quality judgments as DPO training signal — self-distillation, not calibration, and the exact pattern this revision was built to remove. Every VLM check remaining in this pipeline is a content gate (is this safe / high-quality / well-structured), not a source of preference rankings. Preference rankings now come exclusively from real human annotators via Pick-a-Pic v2, HPDv2, DesignSense-10k, and DesignPref.

## Known Fixes Applied Across Revisions

**v16 (no-RLHF-loop restructuring):**
- Removed `s01_6_planner_synthesis.py`, `s07_5_edit_pairs.py`, `s10_5_critic_preference.py`, and `inference/critic.py` entirely, along with their `EXECUTION_ORDER` entries and `models.yaml`'s `critic`/`qwen_image_vae` entries.
- Added `s01_6_preference_pairs.py` and `s08_5_dpo_encoding.py` for real, human-labeled DPO data.
- `RICO core`/`RICO semantic` were configured to scan for loose image files against repos that ship exclusively parquet-embedded images — would have silently fetched **zero records** from the single most foundational UI-domain source (CLAY, Enrico, and Screen2Words all join onto RICO). Fixed with a dedicated `hf_parquet_images` fetch adapter; both repos' distinct image-column names (`"screenshot"` vs. `"image"`) confirmed directly against their live dataset cards and hardcoded rather than left to auto-detection.
- `../../data-forge/data_forge/utils/completeness.py` still required `qwen_image_latent` after that encoding branch was removed — would have flagged every record in the corpus as incomplete, starving Stage 9/10/12 of all input. Fixed.
- `storage.py`'s pre-flight check summed every dataset's raw `expected_record_count` instead of respecting `sample_size` caps — inflated the projected corpus from ~100K-500K to ~26M records / ~33TB, which would false-fail the pre-flight check before Stage 1 ever ran. Fixed via `DatasetSpec.storage_relevant_record_count()`.
- Preference-pair images had zero safety/NSFW classification — Pick-a-Pic v2/HPDv2 are open-T2I-model outputs at scale, a known source of occasional NSFW content even after curation. Fixed: Stage 1.6 now runs the same safety classifier the main manifest uses.
- HPDv2's real schema (a variable-length ranked list, not a fixed two-image table) didn't match the generic `preference_pair` fetcher — would have produced zero pairs. Fixed with a dedicated `hpdv2_ranked_list` adapter.
- `cli.py`'s `--stages` shortcut map still pointed at deleted stage names after the removal. Fixed.
- This document and `README.md` were themselves stale through earlier passes of this same revision — corrected here.

**v14/v15, still in effect:** OCR enrichment's missing config block (and the PII redaction it silently disabled), the never-published Qwen-Image-2.0-VAE reference (moot now — Qwen-Image-Edit-2511's training path is removed entirely, not just re-pointed), PD12M/CC12M's metadata-only ingestion gap, a full-table manifest scan repeated per dataset, an arbitrary Windows MAX_PATH threshold, dead FAISS `nprobe` config, an unclosed `httpx.AsyncClient` in the registry watcher, three UI datasets missing from the domain tagger's source list, two duplicated `StageResult` dataclasses, and a reverted `ShardRouter` "fix" that was actually wrong (see that file's own docstring for why).
