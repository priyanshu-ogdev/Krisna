# Research-paper artifact map

This document is the handoff between the paper specification and the
implementation. A paper claim is publishable only when the corresponding
artifact and receipt exist; a design document or passing unit test is not a
measured result.

| Paper section | Implementation boundary | Required artifact | Current status |
|---|---|---|---|
| Problem/system definition | `docs/RESEARCH_PAPER.md`, `inference/src/krivex_inference/` | Versioned architecture and API contracts | Defined |
| Data construction | `data-forge/`, `training/src/krivex_training/data_forge_bridge/` | Source-disjoint pair manifest, provenance and license receipts | Not realized |
| Planner/Layout IR | `inference/src/krivex_inference/layout_*`, planner backends | Fixed-seed planner and layout evaluation receipt | Contract-tested; not benchmarked |
| Qwen-Image-2.1 training | `training/src/krivex_training/polish/` | Native one-batch GPU train/save/load receipt | Preflight and contract only |
| Inference/locking | `inference/src/krivex_inference/backends/`, `verifiers/` | Native inference and protected-pixel evaluation receipts | Unit-tested; GPU gate pending |
| Evaluation | `docs/architecture/EVALUATION_PROTOCOL.md`, `tests/inference/` | OCR/CER, layout, lock, latency, VRAM, and seed manifests | Protocol defined; measurements pending |
| Ablations | `docs/research/FLOW_MATCHING_UPGRADE_PLAN.md` | Matched seeds, splits, hardware, and baseline outputs | Planned |

## Publication gate

The paper may describe the system and protocol now. It must not report numeric
quality, memory, latency, superiority, or statistical-significance claims until
the corresponding receipt is checked into the external artifact store and its
hash is referenced from an evaluation manifest. Synthetic fallback outputs and
unit-test fixtures remain implementation evidence only.

## Reproduction order

1. Verify source and model revisions with the Data-Forge audit.
2. Materialize and hash the source-disjoint Qwen pair corpus.
3. Run the CPU training preflight and review the redacted command.
4. Run the native GPU one-batch train/save/load/inference gate.
5. Run the fixed evaluation protocol on all baselines and ablations.
6. Link the receipts and exact commit to the paper's results section.
