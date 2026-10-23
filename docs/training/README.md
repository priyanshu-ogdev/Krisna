# Training documentation

This directory is the authoritative training documentation set. It describes
the current executable source tree, its known gaps, and the gated research
architecture. It does not claim that a configuration or launcher is a completed
training run.

## Start here

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — current model/trainability matrix,
   data contracts, agentic image/VL training boundary, research comparisons, and
   the recommended gate sequence.
2. **[TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)** — read-only readiness checks,
   data preparation, the supported Polish launcher, and post-training evaluation.
3. **[Evaluation Harness & Runbook](../../scripts/eval/README.md)** — multi-hardware benchmark evaluation, synthetic testing mode, and automated LaTeX monograph synchronization.
4. **[Evaluation Ingestion Tool](../../scripts/eval/ingest_eval_results.py)** — automated evaluation ingestion, pre-registered hypothesis testing, and cryptographic receipt generation.

## Current training decision

| Component | Current status |
|---|---|
| Planner | Frozen Qwen3.5-9B GGUF/RAG path; no active Planner LoRA training |
| Structure | Deterministic `krivex.layout.v1` baseline; no complete learned Sketch trainer in this checkout |
| Polish Default | Qwen-Image-2.1 supervised DoRA; Enterprise Max ($r=128$, FP32 AdamW, 45GB peak), Enterprise Standard ($r=64$), and Consumer Desktop ($r=64$, NF4) profiles wired |
| Polish Quality | Frozen Qwen-Image-2.1 comparison path |
| DPO | Deferred historical path; not part of the active launcher |

## Source-of-truth checks

Run the read-only audit before preparing a run:

```bash
python scripts/training/audit_training_pipeline.py
```

The audit reports model IDs, missing source/trainer artifacts, configured and
realized dataset counts, and launcher inconsistencies. It does not install
dependencies, download weights, modify data, or start training.

## Important terminology

- **Configured** means a model ID, path, or expected count appears in source.
- **Realized** means the corresponding artifact exists in the current workspace.
- **Validated** means a smoke test or run receipt proves the contract.
- **Measured** means a real run produced hardware and evaluation evidence.

The research papers cited in this section are precedents for specific design
choices only. They do not establish that Krivex has reproduced their scores or
that an external trainer accepts the repository's current flags.
