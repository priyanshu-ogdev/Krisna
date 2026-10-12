# Stage 09 — Planner Review and Critic retirement

## Product decision

The active critique route now calls Planner Review while Planner is resident.
Gemma Critic is no longer required for the product path.

## Review mode

Review should combine:

- current render or render-derived evidence;
- active design constraints;
- verifier results;
- conversation history;
- retrieved human critique examples.

On low-VRAM native PCs, planning remains text-only and Review uses vision only
when a validated mmproj profile fits. OCR, layout, and safety verifiers remain
hard gates rather than being replaced by a generative judgment.

## Validation before deletion

Compare:

1. Gemma Critic + verifiers;
2. Planner Review + verifiers;
3. both together on a larger offline machine.

Measure defect recall, false positives, edit usefulness, human agreement, and
latency. Only after Planner Review passes should archived Critic code be removed
from the registry, factory, setup, and readiness checks.
