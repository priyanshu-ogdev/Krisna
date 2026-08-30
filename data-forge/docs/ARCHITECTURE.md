# Data-Forge Architecture Specification

> **UPGRADED (no-RLHF-loop revision, v16):** the planner (Qwen3.5-9B) and
> Qwen-Image-Edit-2511 now ship **frozen** — RAG + constrained decoding
> for the former, zero-shot ICL + SDEdit for the latter — so `s01_6` is
> now `s01_6_preference_pairs.py` (real human DPO preference data, not
> synthetic planner conversations) and there is no `s07_5`/`Branch 2
> qwen_image_vae` step anymore. A new `s08_5_dpo_encoding.py` encodes
> preference pairs into Z-Image-Turbo's latent space instead. The Gemma-4
> critic tier is a frozen, on-demand product feature only — `s10_5` has
> been removed outright, no AI-judge labeling happens anywhere in this
> pipeline. The narrative below (v15 and earlier) is kept as the
> reasoning trail for *why* each prior fix happened; where it conflicts
> with this note, this note is current.

## Overview

The Data-Forge is a custom, 20-stage Python orchestrator designed to process millions of images into training-ready latents on a single workstation (specifically a 48GB VRAM GPU like an RTX A6000). It replaces manual curation with **VLM-as-Judge** auditing and deterministic logic, enforcing a strict zero-touch automation philosophy (v15 spec — v14 plus dedicated data production for every PRD-trainable component; see `docs/DATA_COMPLETENESS.md`).

## The Orchestrator (Chunk-Based DAG)

Preprocessing 12M images through multiple VLMs (Qwen, VAEs, OCR, Safety) on a single GPU normally suffers from severe PCIe swapping bottlenecks. The Data-Forge solves this by utilizing a **Chunk-Based Directed Acyclic Graph (DAG)**.

1. **Chunking**: The orchestrator splits the entire dataset into chunks (default 10,000 records).
2. **Model Pinning**: A required model (e.g., Tier-1 VLM) is loaded into VRAM.
3. **Execution**: The orchestrator runs all records in the chunk through every stage that requires that specific model.
4. **Teardown**: The model is gracefully unloaded, CUDA cache is cleared, and the next model (e.g., DeepSeek OCR, or the Critic Tier) is loaded.

## Pipeline Stages

| Stage | Name | GPU Model | Purpose |
|---|---|---|---|
| **00** | Manifest Planning | — | Init SQLite DB, storage pre-flight check, read registry watcher report. |
| **01** | Fetch & License | Tier-1 (inline) | Download shards (including a URL-list path for metadata-only sources like PD12M/CC12M — see Data Sources), run License Verification Agent to extract terms and triage. |
| **02** | Dedup | CLIP / FAISS | Exact-hash + semantic near-duplicate removal to save downstream compute. |
| **03** | Quality | Tier-1 | Aesthetic and resolution scoring. |
| **03.5**| PII Scrub | MediaPipe/Regex | Face blurring and sensitive-text redaction. |
| **04** | Safety | Tier-1 | NSFW/Harmful content classification. |
| **04.5**| Escalation | Tier-2 | Second opinion on borderline safety/license records. Routes to `excluded_pending_review`. |
| **05** | Recaption | Tier-1 | Dense structural captioning. |
| **05-OCR** | OCR Enrichment | DeepSeek-OCR | Text-in-image extraction — a distinct, separately-registered stage requiring its own `stages.s05_ocr_enrichment:` config block (previously missing; see Known Fixes below). |
| **05.5**| PII Text Redact | Regex | Redacts sensitive text found by the OCR pass. |
| **06** | Structure | Tier-1 | UI component tree and bounding box JSON extraction. |
| **07** | Routing | — | Domain tagging, ratio enforcement, shard assignment. |
| **08** | Tri-Path Encoding | VAEs / VQ | Encodes Z-Image latents, Qwen-Image-Edit-2511 latents, MaskGIT VQ tokens, and control maps. |
| **09** | Heldout Carve | — | Stratified evaluation set carve-out. |
| **10** | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge rubric evaluation (replaces human spot-checks) on 2-5% of data. |
| **10.5**| Critic Preference | Gemma 4 31B | Bulk critique generation seeding `/ui_critique/*.parquet`. |
| **11** | Registry Watch | — | External cron; polls HF/GitHub for model/dataset updates, writes report. |

## Tri-Path Latent Storage Strategy

To support the Krisna Two-Tier Generation Architecture (Sparse Masked Sketching -> Continuous Flow Polishing) plus the on-demand Critic Tier, the Data-Forge encodes every approved image into six distinct representations:

1. `latents_zimage/` (fp16 `.safetensors`) - Continuous latents for Z-Image-Turbo.
2. `latents_qwenimage/` (fp16 `.safetensors`) - Continuous latents for Qwen-Image-Edit-2511.
3. `vq_tokens_sketch/` (int16 `.pt`) - Discrete codebook indices for MaskGIT/MAR.
4. `control_tokens/` (`.json`) - Structural bounding boxes and Canny edges.
5. `ui_critique/` (`.parquet`) - Scored critiques (UICrit-rubric + Gemma-4-generated), Stage 10.5's output.
6. `preference_pairs/` (`.parquet`) - Reserved for ranked DPO pairs; population is out of this pipeline's scope until the product's own generation loop exists.

## Sub-System Architecture

### 1. State Management (SQLite Manifest)
All state is tracked in a local SQLite database (`manifest.db`). The manifest operates in `WAL` mode with `isolation_level="IMMEDIATE"` and a 5000ms `busy_timeout` to prevent locking issues on Windows filesystems. Every record transition is logged in the `stage_history` table for a complete audit trail. A lightweight, idempotent column-migration step runs on every open, so a `manifest.db` from an older revision picks up newly-added columns automatically.

### 2. Inference Engine Lifecycle
The inference engine (`engine.py`) handles the lifecycle of models.
- **vLLM subprocess**: The Tier-1, Tier-2, and Critic (Gemma 4 31B) models are spawned as HTTP servers via vLLM. On teardown, `psutil` recursively kills all background CUDA worker threads before killing the parent process, ensuring zero VRAM leakage.
- **Transformers/PyTorch**: The CLIP model, VAEs, and VQ tokenizers are loaded natively via PyTorch and cleared using `gc.collect()` and `torch.cuda.empty_cache()`. Each encoder now loads inside its own try/except — one bad model reference (e.g. a repo that was never published) degrades that one encoder's output rather than crashing the whole Stage 8 run.

### 3. VLM-as-Judge & Escalation
Automated VLM judgments replace human validation. The pipeline uses an **Ensemble Disagreement** strategy: if a downstream reasoning check (like the Stage 10 Audit) disagrees with an upstream filter (like Stage 3 Quality or Stage 4 Safety), the record is flagged and escalated. Ambiguous cases are routed to the `excluded_pending_review` bucket instead of halting the pipeline. Stage 10.5's Critic Tier judgments are a separate, additive signal — see `s10_5_critic_preference.py`'s module docstring for the distinction between scored single-image critiques (what this pipeline produces) and ranked DPO preference pairs (what the product's own training loop will need to construct downstream).

## Known Fixes Applied in This Revision

- OCR enrichment (`s05_ocr_enrichment`) had no config entry and silently defaulted to disabled, which also silently disabled all downstream text-PII redaction.
- Stage 8's `qwen_image_vae` pointed at a repo (`Qwen/Qwen-Image-2.0-VAE`) that was never published, crashing the entire encoding phase on first use. Replaced with Qwen-Image-Edit-2511.
- PD12M and CC12M (roughly half the target corpus) were never actually ingested — the fetcher had no URL-list download path for metadata-only sources.
- `/ui_critique/` and `/preference_pairs/` had no implementing code anywhere despite being documented outputs.
- A full-table manifest scan repeated per dataset in Stage 1's license-verification loop, now a scoped, indexed query.
- The Windows MAX_PATH check used an arbitrary root-length threshold rather than the pipeline's actual worst-case path length.
- Dead FAISS `nprobe` config, an unclosed `httpx.AsyncClient` in the registry watcher, three UI datasets missing from the domain tagger's source list, and two duplicated `StageResult` dataclasses were all fixed.
