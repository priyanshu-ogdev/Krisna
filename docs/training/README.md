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
   data preparation, and the one supported Polish launcher.

## Current training decision

| Component | Current status |
|---|---|
| Planner | Frozen Qwen3.5-9B GGUF/RAG path; no active Planner LoRA training |
| Structure | Deterministic `krivex.layout.v1` baseline; no complete learned Sketch trainer in this checkout |
| Polish Default | Intended Qwen-Image-2.1 supervised LoRA; pair preparation exists, external trainer/API still requires validation |
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
