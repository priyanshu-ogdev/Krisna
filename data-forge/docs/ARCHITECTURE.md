# Data-Forge Architecture Specification

## Overview

The Data-Forge is a custom, 12-stage Python orchestrator designed to process millions of images into training-ready latents on a single workstation (specifically a 48GB VRAM GPU like an RTX A6000). It replaces manual curation with **VLM-as-Judge** auditing and deterministic logic, enforcing a strict zero-touch automation philosophy (v13 spec).

## The Orchestrator (Chunk-Based DAG)

Preprocessing 12M images through multiple VLMs (Qwen, VAEs, OCR, Safety) on a single GPU normally suffers from severe PCIe swapping bottlenecks. The Data-Forge solves this by utilizing a **Chunk-Based Directed Acyclic Graph (DAG)**.

1. **Chunking**: The orchestrator splits the entire dataset into chunks (default 10,000 records).
2. **Model Pinning**: A required model (e.g., Tier-1 VLM) is loaded into VRAM.
3. **Execution**: The orchestrator runs all records in the chunk through every stage that requires that specific model.
4. **Teardown**: The model is gracefully unloaded, CUDA cache is cleared, and the next model (e.g., DeepSeek OCR) is loaded.

## Pipeline Stages

| Stage | Name | GPU Model | Purpose |
|---|---|---|---|
| **00** | Manifest Planning | — | Init SQLite DB, storage pre-flight check, read registry watcher report. |
| **01** | Fetch & License | Tier-1 (inline) | Download shards, run License Verification Agent to extract terms and triage. |
| **02** | Dedup | CLIP / FAISS | Exact-hash + semantic near-duplicate removal to save downstream compute. |
| **03** | Quality | Tier-1 | Aesthetic and resolution scoring. |
| **03.5**| PII Scrub | MediaPipe/Regex | Face blurring and sensitive-text redaction. |
| **04** | Safety | Tier-1 | NSFW/Harmful content classification. |
| **04.5**| Escalation | Tier-2 | Second opinion on borderline safety/license records. Routes to `excluded_pending_review`. |
| **05** | Recaption + OCR | Tier-1 -> OCR | Dense structural captioning + text-in-image extraction. |
| **06** | Structure | Tier-1 | UI component tree and bounding box JSON extraction. |
| **07** | Routing | — | Domain tagging, ratio enforcement, shard assignment. |
| **08** | Tri-Path Encoding | VAEs / VQ | Encodes Z-Image latents, Qwen-Image latents, MaskGIT VQ tokens, and control maps. |
| **09** | Heldout Carve | — | Stratified evaluation set carve-out. |
| **10** | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge rubric evaluation (replaces human spot-checks) on 2-5% of data. |
| **11** | Registry Watch | — | External cron; polls HF/GitHub for model/dataset updates, writes report. |

## Tri-Path Latent Storage Strategy

To support the Krisna Two-Tier Generation Architecture (Sparse Masked Sketching -> Continuous Flow Polishing), the Data-Forge encodes every approved image into four distinct representations during Stage 8:

1. `latents_zimage/` (fp16 `.safetensors`) - Continuous latents for Z-Image-Turbo.
2. `latents_qwenimage/` (fp16 `.safetensors`) - Continuous latents for Qwen-Image-2.0.
3. `vq_tokens_sketch/` (int16 `.pt`) - Discrete codebook indices for MaskGIT/MAR.
4. `control_tokens/` (`.json`) - Structural bounding boxes and Canny edges.

## Sub-System Architecture

### 1. State Management (SQLite Manifest)
All state is tracked in a local SQLite database (`manifest.db`). The manifest operates in `WAL` mode with `isolation_level="IMMEDIATE"` and a 5000ms `busy_timeout` to prevent locking issues on Windows filesystems. Every record transition is logged in the `stage_history` table for a complete audit trail.

### 2. Inference Engine Lifecycle
The inference engine (`engine.py`) handles the lifecycle of models.
- **vLLM subprocess**: The Tier-1 and Tier-2 models are spawned as HTTP servers via vLLM. On teardown, `psutil` recursively kills all background CUDA worker threads before killing the parent process, ensuring zero VRAM leakage.
- **Transformers/PyTorch**: The CLIP model, VAEs, and VQ tokenizers are loaded natively via PyTorch and cleared using `gc.collect()` and `torch.cuda.empty_cache()`.

### 3. VLM-as-Judge & Escalation
Automated VLM judgments replace human validation. The pipeline uses an **Ensemble Disagreement** strategy: if a downstream reasoning check (like the Stage 10 Audit) disagrees with an upstream filter (like Stage 3 Quality or Stage 4 Safety), the record is flagged and escalated. Ambiguous cases are routed to the `excluded_pending_review` bucket instead of halting the pipeline.
