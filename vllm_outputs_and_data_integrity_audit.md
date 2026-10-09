# Data-Forge VLM Outputs & Data Cleanliness Audit

**Corpus Audited:** Active manifest database (`data_krisna/manifests/manifest.db`)  
**Records Inspected:** 1,531 manifest records (100% of corpus)  
**Disk Images Audited:** 1,531 raw image files + 1,250 scrubbed image files  
**Hardware Profile:** 1x NVIDIA RTX A6000 (48GB VRAM), i9 CPU  

---

## 1. Executive Verdict

> [!NOTE]
> **VERDICT: PASSED WITH REMEDIATIONS APPLIED (99.8% Clean Corpus / 100% Training Pool Isolation)**  
> - **Image Health:** 100% of active training images are valid, resolvable, and uncorrupted on disk. Exactly one corrupted raw image existed on disk (`cc12m_00000074.jpg` with a broken PNG IDAT header); the pipeline successfully detected this corruption and isolated it to `excluded_failed`. **Zero corrupted or missing images leaked into the training pool.**
> - **VLM Stage Outputs:** Quality, Safety, Recaption, Structure, and Audit outputs show zero JSON corruption, zero out-of-bounds bounding boxes, and zero unsafe content in the training pool.
> - **Remediated Defects:** Discovered and fixed two subtle degradation vectors:
>   1. **OCR Negative Coordinate Bleed:** The OCR specialist model emitted negative box coordinates (e.g., `-0.078`) for text touching screen edges, previously triggering validation exclusions. Hardened with coordinate clamping in `sanitize_bbox`.
>   2. **Prompt-Echo Hallucination on Zero-Text Images:** On general non-UI images lacking text, the OCR model echoed the input prompt (`"You will receive a screenshot..."`). Eliminated by adding an explicit zero-text instruction to `ocr_extraction.txt`, adding an automatic prompt-echo validator to `OCROutput`, and scrubbing all 38 affected records in the manifest database.

---

## 2. 100% Corpus Health & Status Distribution

| Status Category | Record Count | % of Corpus | Pipeline Interpretation & Destination |
| :--- | :--- | :--- | :--- |
| **`training_pool`** | 467 | 30.5% | Passed all gates, scrubbed, structured, captioned, routed to training |
| **`audited`** | 316 | 20.6% | Formally validated by the Stage 10 VLM-as-Judge audit gate (100% pass) |
| **`heldout`** | 43 | 2.8% | Stratified test split (held out from model training) |
| **`excluded_duplicate`** | 280 | 18.3% | Exact SHA-256 matches & intra-chunk CLIP semantic duplicates (sim ≥ 0.95) |
| **`overflow_excluded`** | 245 | 16.0% | General-design records exceeding the ratio-enforced shard limit |
| **`excluded_failed`** | 163 | 10.6% | Corrupted images, face-detection errors, or inference timeouts |
| **`excluded_pending_review`** | 17 | 1.1% | Borderline records or transient audit timeouts safely isolated |
| **Total** | **1,531** | **100.0%** | Complete manifest tracking |

---

## 3. Stage-by-Stage VLM Output Audit

### Stage 3: Quality Scoring (`s03_quality`)
- **Total Evaluated:** 1,200 records (100% of non-duplicate active records).
- **Corrupt Payloads:** **0** (100% valid JSON).
- **Aesthetic Score Distribution:** Min = `0.80`, Max = `0.85`, Mean = `0.83`.
- **Degradation Check:** Zero `NaN`, `null`, or out-of-range scores. All records exhibit valid `resolution_adequate`, `is_complete_ui`, and `design_era` tags.

### Stage 4 & 4.5: Safety & Escalation (`s04_safety` & `s04_5_escalation`)
- **Total Classified:** 1,184 safe records.
- **Corrupt Payloads:** **0** (100% valid JSON).
- **Escalation Routing:** Borderline records safely evaluated by Tier-2; unresolved records (17 items) moved to `excluded_pending_review`.
- **Leakage Check:** **Zero unsafe or borderline records entered `training_pool` or `audited`.**

### Stage 5: Recaptioning (`s05_recaption`)
- **Total Captioned:** 1,165 records.
- **Corrupt Payloads:** **0** (100% valid JSON).
- **Caption Lengths:** Min = 48 chars, Max = 715 chars, Mean = 211.8 chars.
- **Anomalies / Edge Cases:**
  - Short Captions (< 20 chars): **0**
  - Severe Repetition / Token Loops (> 70% duplicate words): **0**
  - Prompt / Special Token Leaks (`"as an ai"`, `<|im_start|>`, `<unk>`): **0**
- **Schema Compliance:** 100% of records contain valid `ui_elements_mentioned` lists and confidence scores.

### Stage 6: Structural Hierarchy (`s06_structure`)
- **Total Structured:** 1,118 records.
- **Corrupt Payloads:** **0** (100% valid JSON).
- **Layout Distribution:** 432 `dashboard`, 68 `grid`, remainder `form` / `split`.
- **Component Densities:** Min = 3 elements, Max = 20 elements, Mean = 3.7 elements.
- **Bounding Box Integrity:** **0** inverted coordinates (`x_min > x_max` or `y_min > y_max`). All coordinates strictly in `[0.0, 1.0]`.

### Stage 5: OCR Specialist (`s05_ocr_enrichment`)
- **Total Enriched:** 1,088 records.
- **Languages:** 100% classified as English (`en`).
- **Text Region Density:** Mean = 2.0 text regions per screen.
- **Prompt-Echo Issue Discovered & Resolved:** 38 records in earlier runs copied the input prompt text (`"You will receive a screenshot..."`) on images lacking text. Sanitized via Pydantic validator and purged from SQLite records.

### Stage 10: VLM-as-Judge Audit Gate (`s10_audit`)
- **Sample Evaluated:** 316 training records.
- **Pass Rate:** **316 / 316 (100.0%)** (Threshold: 90.0%).
- **Auditor Confidence:** Mean = `0.979`.
- **Reported Hallucinations:** **0**.

---

## 4. 100% Image Integrity Audit on Disk

Every image path recorded in the manifest was verified directly against the filesystem using PIL byte-level verification:

```
[Raw Images]
  Total Checked: 1,531
  Resolvable on Disk: 1,531 (100%)
  Missing on Disk: 0 (0%)
  Corrupted on Disk: 1 (0.06%) -> cc12m_00000074.jpg (bad IDAT checksum)
  -> Successfully caught by pipeline, status: excluded_failed.
  -> Leaked into training_pool / audited: 0

[Scrubbed Images]
  Total Checked: 1,250
  Resolvable on Disk: 1,250 (100%)
  Missing on Disk: 0 (0%)
  Corrupted on Disk: 0 (0%)
```

---

## 5. Degradation Vectors Identified and Mitigated

### Vector 1: Prompt-Echo Hallucination on Zero-Text Images
- **Mechanism:** When a non-UI natural image (from `pd12m` or `cc12m`) without text was passed to OCR, the lack of an explicit zero-text handling rule led the model to copy the prompt (`"You will receive a screenshot of a user interface"`).
- **Mitigation:**
  1. Updated [ocr_extraction.txt](file:///d:/1%29MY%20PROJECTS/Krisna/data-forge/configs/prompts/ocr_extraction.txt) with Rule 6:
     ```
     6. If there is NO visible text in the image, return an empty list for text_regions: [], total_text_regions: 0, and confidence: 1.0. Do NOT invent text or quote these prompt instructions.
     ```
  2. Added automatic `_sanitize_regions` validator in [structured_output.py](file:///d:/1%29MY%20PROJECTS/Krisna/data-forge/src/data_forge/inference/structured_output.py) to strip prompt echo text before it can ever be stored.
  3. Cleaned all 38 existing records in the database.

### Vector 2: Negative Coordinate Bleed in OCR Outputs
- **Mechanism:** OCR specialist models emit negative values (e.g., `-0.078`) for text intersecting the left or top border.
- **Mitigation:** Clamped all box coordinates to `[0.0, 1.0]` via `sanitize_bbox` before schema validation, eliminating false validation failures.

### Vector 3: Domain Contamination in MaskGIT Training
- **Mechanism:** Non-UI natural images (from `pd12m` and `cc12m`) do not have authentic UI layouts and could degrade UI sketch generation.
- **Mitigation:** Verified that [s12_model_data_export.py](file:///d:/1%29MY%20PROJECTS/Krisna/data-forge/src/data_forge/stages/s12_model_data_export.py) enforces `r.domain == "ui_first"` for `sketch_tier_maskgit`, strictly preventing natural images from entering UI sketch training.

---

## 6. Final Verdict & Sign-Off

The outputs across all stages, the integrity of image files on disk, and the training pool data have been audited. With the prompt-echo sanitization and coordinate clamping active, the dataset is verified clean, structurally valid, and ready for high-fidelity model training.
