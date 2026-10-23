# scripts/eval/ — Post-Training Multi-Hardware Evaluation & Paper Synchronization

This directory contains the authoritative automated evaluation harness and research monograph synchronization pipeline for the **Krivex Agentic UI Design System**.

---

## 1. Overview & Capabilities

The evaluation suite bridges post-training checkpoint validation, multi-hardware benchmark testing, and automated scientific publication:
1. **Multi-Metric Quantitative Benchmarks**: Computes distribution fidelity (KID, CMMD), perceptual quality (LPIPS, DINOv2), spatial alignment (OpenCV Layout-IoU), typographic legibility (EasyOCR CER/WER), aesthetic quality (LAION Aesthetic), and turn latency.
2. **Multi-Hardware Profiles**: Evaluates configurations across Enterprise Max (48GB A6000), Mid-Tier (24GB RTX 3090/4090), and Consumer Desktop (12GB RTX 4070 NF4 QDoRA).
3. **Multi-Format Publication Exports**: Emits machine-readable JSON (`eval_comparison.json`), documentation Markdown (`eval_comparison.md`), and publication-grade LaTeX booktabs code (`eval_comparison_latex.tex`).
4. **Automated Hypothesis Verification**: Formally tests the 7 pre-registered OSF hypotheses (H1–H7) against declared tolerance bounds.
5. **Cryptographic Provenance Receipts**: Emits `evaluation_receipt.json` documenting Git commit SHAs, timestamps, profile metrics, and SHA-256 integrity checksums.
6. **Automatic Research Monograph Synchronization**: Directly updates empirical evaluation tables (Tables 11.1–11.6) in `docs/RESEARCH_PAPER.md`.

---

## 2. Tools & Scripts

### 2.1 `run_evaluation.py` — Benchmark Runner

Executes model generation or synthetic generation passes across held-out splits, running the automated `VerifierStack` and image-quality metrics.

#### Command-Line Arguments:
- `--profile`: Hardware profile to evaluate (`48gb`, `24gb`, `12gb`). Default: `48gb`.
- `--checkpoint`: Path to fine-tuned DoRA adapter directory (e.g., `checkpoints/qwen_image21_lora_a6000_max`).
- `--reference-dir`: Directory containing ground-truth human UI screenshots (`data/reference_uis`).
- `--wireframe-dir`: Directory containing held-out wireframe condition images (`data/held_out_wireframes`).
- `--prompt-list`: JSON file with held-out design prompts (`data/eval_prompts.json`).
- `--output-dir`: Destination folder for evaluation artifacts. Default: `eval_results`.
- `--synthetic`: **Fast Dry-Run Mode**. Bypasses unquantized 14GB base model weights, synthesizing test images to verify the complete verification, scoring, and export pipeline without requiring high-end GPU allocations.
- `--num-samples`: Number of evaluation samples to process. Default: `100` (`5` recommended for quick dry-runs).

#### Execution Examples:

```bash
# Production Server Evaluation (RTX A6000 48GB):
python scripts/eval/run_evaluation.py \
  --profile 48gb \
  --checkpoint checkpoints/qwen_image21_lora_a6000_max \
  --reference-dir data/reference_uis \
  --wireframe-dir data/held_out_wireframes \
  --prompt-list data/eval_prompts.json \
  --output-dir eval_results

# Fast Offline Smoke Test / CI Dry-Run:
python scripts/eval/run_evaluation.py --synthetic --num-samples 5 --output-dir eval_results
```

---

### 2.2 `ingest_eval_results.py` — Ingestion, Hypothesis Verification & Monograph Sync

Ingests structured benchmark results, checks hypothesis bounds, issues cryptographic provenance receipts, and updates academic tables.

#### Pre-Registered Hypotheses (PRD §8 & Research Paper §11):
- **H1 (Layout Fidelity)**: $\text{Layout-IoU} \ge 0.880$ (Target: $0.912$).
- **H2 (Zero-Eviction Serving Latency)**: Turn latency $\le 1.200\text{s}$ with $T_{\text{swapping}} \equiv 0.000\text{s}$.
- **H3 (Region Lock Retention)**: $S_{\text{lock}} \ge 0.990$ with zero spatial drift.
- **H4 (Typographic Legibility)**: Character Error Rate $\text{CER} \le 0.080$ (Target: $0.042$).
- **H5 (Visual Quality Improvement)**: $\text{KID} \le 1.800 \times 10^3$ and $\text{LPIPS} \le 0.160$.
- **H6 (Aesthetic Score)**: LAION Aesthetic Score $\ge 7.800 / 10.0$.
- **H7 (Consumer Memory Ceiling)**: Peak VRAM $\le 11.5\text{ GB}$ on Consumer 12GB NF4 profile.

#### Command-Line Arguments:
- `--input`: Path to `eval_comparison.json` benchmark results.
- `--paper-path`: Path to `docs/RESEARCH_PAPER.md` monograph.
- `--dry-run`: Validates hypotheses and displays table updates without modifying files.
- `--apply`: Automatically writes updated Markdown tables to `docs/RESEARCH_PAPER.md`.
- `--generate-template`: Generates a synthetic template JSON file for testing.

#### Execution Examples:

```bash
# 1. Verify hypotheses against empirical results (read-only audit):
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --dry-run

# 2. Ingest results and update research paper monograph:
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --paper-path docs/RESEARCH_PAPER.md \
  --apply
```

---

## 3. Generated Artifacts Reference

| Artifact | Format | Description |
|---|---|---|
| `eval_comparison.json` | JSON | Authoritative quantitative metric dictionary across all evaluated profiles. |
| `eval_comparison.md` | Markdown | Rendered GitHub-flavored Markdown table for documentation and PR review. |
| `eval_comparison_latex.tex` | LaTeX | Booktabs-formatted LaTeX table with standard standard errors for publication. |
| `evaluation_receipt.json` | JSON | Cryptographic receipt with SHA-256 hashes, Git commit SHA, and test timestamps. |
