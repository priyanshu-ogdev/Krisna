# Krisna Data-Forge: System Architecture (v13)

The **Data-Forge** is a fully automated, zero-touch data ingestion pipeline designed for the Krisna conversational AI design agent. It processes millions of raw web-scraped images into training-ready latents (Tri-Path encoding) on a single workstation (e.g., RTX A6000 48GB GPU + 128GB RAM). 

This architecture implements the **v13 Zero-Touch Specification**, strictly replacing human-in-the-loop manual spot-checks with deterministic logic and Resident VLM-as-Judge reasoning models.

---

## 1. Zero-Touch Automation Philosophy

The data-forge operates under a strict rule: **No manual data processing or curation.**
Every step is automated into one of two buckets:

1. **Deterministic Logic**: Tasks with a defined right answer (deduplication thresholds, schema conformance, shard routing, storage mathematics) are hard-coded in Python.
2. **Resident VLM-as-Judge**: Tasks requiring interpretation or rubric evaluation (captioning, UI extraction, safety adjudication, spot-checking) are routed to a resident open-weight multimodal model.

**The Human Exception**: The only manual interventions allowed are:
* Final legal sign-off on ambiguous licenses logged to the `excluded_pending_review` database.
* Product-strategy decisions (e.g., UI-first vs. General-Design sampling ratios).

---

## 2. Decoupled Compute Lanes (GPU vs. CPU)

To prevent the pipeline's tooling model from bottlenecking the product's interactive serving model, inference is fundamentally decoupled into independent hardware lanes:

| Role | Model | Compute Target | Footprint |
| :--- | :--- | :--- | :--- |
| **Product Planner (Unchanged)** | Qwen3.5-9B, BF16 LoRA | GPU-resident (VRAM) | 18GB VRAM |
| **Tier-1 Bulk Engine** | Qwen3.6-35B-A3B-Instruct (Q4 GGUF) | CPU-resident (System RAM) | 21-28GB RAM |
| **Tier-2 Escalation Engine** | Qwen3.6-27B (Q4 GGUF) | CPU-resident (System RAM) | 17-35GB RAM |
| **Dedicated OCR Pass** | DeepSeek-OCR 2 / Dots.OCR | GPU / CPU Hybrid | Small |

**Why System RAM?** By utilizing the workstation's 128GB of System RAM for the Qwen3.6 MoE architecture (only 3B active parameters per token), the pipeline can run continuous verification and auditing locally *without* competing for the 48GB GPU VRAM required by the Tri-Path VAE/VQ encoders.

---

## 3. The 12-Stage Chunk-Based DAG

The orchestrator splits the dataset into chunks (e.g., 10,000 records) to prevent memory fragmentation. Each chunk traverses the following stages:

| Stage | Name | Compute Engine | Purpose |
|---|---|---|---|
| **00** | Manifest Planning | SQLite | Storage pre-flight check, database initialization. |
| **01** | Fetch & License | Tier-1 | Downloads shards. `License Verification Agent` extracts terms. |
| **02** | Dedup | FAISS-GPU | Exact-hash and semantic near-duplicate removal. |
| **03** | Quality | Tier-1 | Aesthetic and resolution scoring. |
| **03.5** | PII Scrub | MediaPipe | Redacts faces and sensitive text on UI screens. |
| **04** | Safety | Tier-1 | NSFW and harmful content classification. |
| **04.5** | Escalation | Tier-2 | Second-opinion on borderline safety/license records. |
| **05** | Recaption + OCR | Tier-1 + DeepSeek | Dense captioning + Text-in-image extraction. |
| **06** | Structure | Tier-1 | UI component tree and bounding box JSON extraction. |
| **07** | Routing | Logic | Domain tagging, ratio enforcement, shard assignment. |
| **08** | Tri-Path Encoding | VAEs / VQ (GPU) | **Critical:** Encodes latents for Sketch/Polish tiers. |
| **09** | Heldout Carve | Logic | Stratified evaluation set carve-out. |
| **10** | Audit Pass | Tier-1 + Tier-2 | Automated VLM spot-check on 2-5% of the data. |
| **11** | Registry Watch | External Cron | Polls HF/GitHub for new SOTA models and datasets. |

---

## 4. State Management: The SQLite Manifest

The `manifest.db` is the heart of the pipeline, providing ACID-compliant tracking for millions of records. 

* **Concurrency**: Implements `WAL` (Write-Ahead Logging) mode and `isolation_level="IMMEDIATE"` to prevent locking issues on Windows filesystems during high-throughput multiprocessing.
* **Non-Blocking Triage**: Rather than halting the pipeline on failed checks (e.g., ambiguous licenses or failing audit rubrics), records are updated to `status: excluded_pending_review` and bypassed by downstream stages.

---

## 5. Tri-Path Latent Storage Strategy

To support Krisna's interactive workflow, approved images are converted into **four distinct representations** (Stage 8) before training:

1. `latents_zimage/`: fp16 `.safetensors` for Z-Image-Turbo (Continuous Flow-Matching).
2. `latents_qwenimage/`: fp16 `.safetensors` for Qwen-Image-2.0 (High-Fidelity rendering).
3. `vq_tokens_sketch/`: int16 `.pt` codebook indices for MaskGIT/MAR (Sparse iterative layout).
4. `control_tokens/`: `.json` files mapping bounding boxes and Canny edges.

*Storage checks are strictly enforced by the `StorageManager` to preempt `MAX_PATH` errors and `ENOSPC` crashes on Windows.*
