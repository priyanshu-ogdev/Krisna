# Krisna Data-Forge: System Architecture (v14)

The **Data-Forge** is a fully automated, zero-touch data ingestion pipeline designed for the Krisna conversational AI design agent. It processes millions of raw web-scraped images into training-ready latents (Tri-Path encoding) on a single workstation (e.g., RTX A6000 48GB GPU + 128GB RAM). 

This architecture implements the **v15 Zero-Touch Specification** (v14 plus dedicated data production for every PRD-trainable component — see `data-forge/docs/DATA_COMPLETENESS.md` for the full model-to-data trace), strictly replacing human-in-the-loop manual spot-checks with deterministic logic and Resident VLM-as-Judge reasoning models.

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
| **Critic Tier** *(new)* | Gemma 4 31B Dense (NF4) | GPU-resident, on-demand only | ~17GB VRAM |

**Why System RAM?** By utilizing the workstation's 128GB of System RAM for the Qwen3.6 MoE architecture (only 3B active parameters per token), the pipeline can run continuous verification and auditing locally *without* competing for the 48GB GPU VRAM required by the Tri-Path VAE/VQ encoders. The Critic Tier is the one exception deliberately kept GPU-resident rather than CPU-offloaded — it's a dense 31B model (no MoE-style sparse activation to exploit on CPU), loaded on-demand for its own dedicated vLLM session (Phase 6), never held simultaneously with the Tier-1/Tier-2 engines.

---

## 3. The 17-Stage Chunk-Based DAG

The orchestrator splits the dataset into chunks (e.g., 10,000 records) to prevent memory fragmentation. Each chunk traverses the following stages. Note that `05` and `05-OCR` are two distinct, separately-registered stages — a config gap that previously left OCR (and, downstream, all text-based PII redaction) silently disabled has since been fixed; see §6.

| Stage | Name | Compute Engine | Purpose |
|---|---|---|---|
| **00** | Manifest Planning | SQLite | Storage pre-flight check, database initialization. |
| **01** | Fetch & License | Tier-1 | Downloads shards (including a URL-list path for metadata-only sources like PD12M/CC12M). `License Verification Agent` extracts terms. |
| **02** | Dedup | FAISS-GPU | Exact-hash and semantic near-duplicate removal. |
| **03** | Quality | Tier-1 | Aesthetic and resolution scoring. |
| **03.5** | PII Scrub | MediaPipe | Redacts faces and sensitive text on UI screens. |
| **04** | Safety | Tier-1 | NSFW and harmful content classification. |
| **04.5** | Escalation | Tier-2 | Second-opinion on borderline safety/license records. |
| **05** | Recaption | Tier-1 | Dense structural captioning. |
| **05-OCR** | OCR Enrichment | DeepSeek-OCR | Text-in-image extraction — a distinct registered stage (`s05_ocr_enrichment`), gated by `s05_recaption`'s config but requiring its own `stages.s05_ocr_enrichment:` block to run at all. |
| **05.5** | PII Text Redact | Regex | Redacts sensitive text found by the OCR pass. |
| **06** | Structure | Tier-1 | UI component tree and bounding box JSON extraction. |
| **07** | Routing | Logic | Domain tagging, ratio enforcement, shard assignment. |
| **08** | Tri-Path Encoding | VAEs / VQ (GPU) | **Critical:** Encodes latents for Sketch/Polish tiers (Z-Image-Turbo VAE + Qwen-Image-Edit-2511 VAE + VQ + control maps). |
| **09** | Heldout Carve | Logic | Stratified evaluation set carve-out. |
| **10** | Audit Pass | Tier-1 + Tier-2 | Automated VLM spot-check on 2-5% of the data. |
| **10.5** | Critic Preference *(new)* | Gemma 4 31B | Bulk critique generation seeding `/ui_critique/*.parquet` — the DPO training signal input this pipeline was previously missing entirely (no code path wrote Parquet at all until this revision). |
| **11** | Registry Watch | External Cron | Polls HF/GitHub for new SOTA models and datasets. |

---

## 4. State Management: The SQLite Manifest

The `manifest.db` is the heart of the pipeline, providing ACID-compliant tracking for millions of records. 

* **Concurrency**: Implements `WAL` (Write-Ahead Logging) mode and `isolation_level="IMMEDIATE"` to prevent locking issues on Windows filesystems during high-throughput multiprocessing.
* **Non-Blocking Triage**: Rather than halting the pipeline on failed checks (e.g., ambiguous licenses or failing audit rubrics), records are updated to `status: excluded_pending_review` and bypassed by downstream stages.
* **Schema migrations**: a lightweight, idempotent `ALTER TABLE ... ADD COLUMN` migration step now runs on every `Manifest.__init__`, so a `manifest.db` created by an older revision picks up new columns (e.g. `critique_output_json`, added this revision) automatically rather than raising `no such column` the first time something writes to it.

---

## 5. Tri-Path Latent Storage Strategy

To support Krisna's interactive workflow, approved images are converted into **six distinct representations** before training: four in Stage 8, two more (Parquet) from Stages 10 and 10.5.

1. `latents_zimage/`: fp16 `.safetensors` for Z-Image-Turbo (Continuous Flow-Matching).
2. `latents_qwenimage/`: fp16 `.safetensors` for Qwen-Image-Edit-2511 (High-Fidelity edit-conditioned rendering).
3. `vq_tokens_sketch/`: int16 `.pt` codebook indices for MaskGIT/MAR (Sparse iterative layout).
4. `control_tokens/`: `.json` files mapping bounding boxes and Canny edges.
5. `ui_critique/`: `.parquet` — scored critiques (UICrit-rubric + Gemma-4-generated), the DPO alignment training signal.
6. `preference_pairs/`: `.parquet` — reserved for ranked A/B DPO pairs once the product's own generation loop exists; see `s10_5_critic_preference.py`'s module docstring for why constructing genuine pairs is out of this pipeline's scope.

*Storage checks are strictly enforced by the `StorageManager` to preempt `MAX_PATH` errors and `ENOSPC` crashes on Windows — the MAX_PATH check now computes the pipeline's actual worst-case nested path length against the real 260-character limit, rather than an arbitrary root-length threshold.*

## 6. Bug Fixes Applied in This Revision

A full audit surfaced several bugs, some severe enough to silently defeat core guarantees (PII redaction, corpus ingestion volume, alignment-training data availability). Summarized here; full reasoning is in code comments at each fix site:

- OCR enrichment (and everything downstream of it, including all text-PII redaction) was silently disabled by a missing config block.
- The entire Tri-Path Encoding stage crashed on first use due to a VAE model reference (`Qwen-Image-2.0-VAE`) that was never actually published.
- PD12M and CC12M — roughly half the target corpus by record count — were never actually ingested; the fetcher had no path for metadata-only, URL-based sources.
- `/ui_critique/` and `/preference_pairs/` were documented pipeline outputs with zero implementing code; `pandas`/`pyarrow` weren't even dependencies.
- A full-table manifest scan repeated once per dataset in the license-verification loop — a real bottleneck at this pipeline's target scale, now a properly indexed, scoped query.
- Three real UI datasets (`enrico`, `screen2words`, `uicrit`) were missing from the domain tagger's source list.
- An unclosed `httpx.AsyncClient` in the registry's HF update checker leaked a connection on every call.
- Two independently-defined, silently-drifting `StageResult` dataclasses consolidated to one.
