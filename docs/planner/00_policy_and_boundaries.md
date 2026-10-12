# Stage 00 — policy and boundaries

## Decision

The Planner is frozen. Production adaptation is limited to:

- prompt/context construction;
- UICrit retrieval;
- constrained decoding;
- inference-time quantization/calibration;
- runtime and lifecycle changes.

The old LoRA training implementation is reference-only. The supported launcher
now exits rather than starting a training run.

## Boundaries

Planner artifacts belong under `models/planner/`. Training outputs for Sketch,
Polish, and DPO remain in their existing directories. A Planner artifact must
not be consumed as a trainable adapter by inference.

## Gate

- no supported command starts Planner training;
- model manifest records source revision and SHA256;
- startup rejects missing or mismatched artifacts;
- docs and setup scripts describe the Planner as frozen.

## Remaining work

Move deprecated Planner source/configuration into an explicit reference/archive
area after checking downstream imports and tests. Do not delete it silently:
historical code is useful for comparison and migration.
