# Inference production stack

Status: implementation-aligned architecture review, 2026-10-14

This document separates what is implemented from what still requires a native
model receipt. It is the operational companion to `docs/RESEARCH_PAPER.md`.

## Runtime planes

```text
Tauri/Rust control plane
  ├─ revision-checked commands
  ├─ bounded refinement budget
  ├─ path-safe local IPC
  └─ lightweight lifecycle/status controls
          │ HTTP on loopback
FastAPI/Python model plane
  ├─ DesignStateStore + session locks
  ├─ SwapOrchestrator + VRAM/RAM admission
  ├─ Planner → Layout IR → wireframe
  ├─ Qwen-Image-2.1 image-conditioned render
  └─ verifier evidence + safety gate
          │ explicit proxy routes / SSE logs
Node/Tauri frontend
  ├─ setup/download status
  ├─ real readiness and residency telemetry
  ├─ actual finalized image bytes
  └─ low-cost Canvas2D effects, paused when hidden
```

## Fail-closed contracts

### Real inference

- `KRISNA_USE_REAL_BACKENDS=1` must never synthesize a replacement image.
- Missing `torch`, `diffusers`, CUDA, Qwen pipeline class, adapter manifest,
  conditioning image, or generation output is an error.
- Synthetic output is allowed only in explicit test mode and carries
  `runtime_mode: synthetic` and `synthetic: true`.
- The renderer does not claim verifier measurements. `VerifierStack` attaches
  post-generation evidence; safety is a gate.
- A Default LoRA is accepted in strict real mode only when its schema-v2
  manifest names the Qwen base/pipeline, dataset-manifest hash, and target
  modules. This keeps training and serving on the same artifact contract.
- The loaded `QwenImage21Pipeline.__call__` signature is authoritative. Strict
  mode does not forward legacy `strength`, custom ODE, or invented guidance
  controls. Receipts record the live scheduler and actual timesteps.
- Polish Quality is frozen and rejects adapter injection. Reproducible
  experiments can disable product-tier fallback so OOM cannot silently change
  the evaluated model.

### Readiness

- `/health` is liveness and exposes `ready` plus `backend_mode`.
- `/ready` is true only after Planner and Layout are admitted and resident.
- `/orchestrator/status` reports both declared ledgers and live hardware probes.

### State and transport

- Rust rejects stale revisions before making a model request.
- Rust validates session IDs before interpolating them into HTTP paths.
- The Node proxy exposes explicit routes rather than a blanket proxy.
- Browser checkpoint paths are passed as serialized data, not interpolated
  executable strings.

## Lightweight effects policy

The UI keeps the product's gradient, glow, flow-field, and wireframe effects,
but treats them as instrumentation rather than model output. Canvas loops stop
when the document is hidden; CSS honors `prefers-reduced-motion`; no new runtime
framework or rendering dependency is required. This preserves a small install
surface and avoids spending GPU/CPU resources on effects while inference is
active.

## SOTA research gates

The following are research gates, not claims that the current checkout has
passed them:

1. Pin the exact Qwen model and Diffusers pipeline class; run a one-batch native
   image-conditioned smoke test.
2. Build source-identity-disjoint condition/target manifests with hashes,
   verified licenses, immutable revisions, and exclusion receipts.
3. Compare deterministic Layout IR against a learned layout ablation using
   layout IoU, text-region OCR, lock preservation, and user preference scores.
4. Measure cold/warm latency, peak VRAM, host RAM under offload, and OOM
   fallback frequency on the declared deployment hardware.
5. Evaluate TASTE and other held-out data without training contamination; report
   confidence intervals and failure cases rather than only means.
6. Enable preference alignment only after the supervised Qwen LoRA passes
   save/load/inference parity and the preference source is human-labeled and
   license-cleared.

## Research-informed design decisions

### IEEE TVCG: PromptMagician

Feng et al., *PromptMagician: Interactive Prompt Engineering for Text-to-Image
Creation*, IEEE Transactions on Visualization and Computer Graphics (2023),
DOI: <https://doi.org/10.1109/TVCG.2023.3327168>, studies interactive prompt
engineering rather than treating one prompt as a complete specification.
The relevant design lesson for Krisna is to preserve an editable, inspectable
prompt/state surface. We therefore keep the effective prompt, constraint delta,
Layout IR revision, seed, and verifier receipt as separate artifacts. We do not
replace Layout IR with prompt search: prompt optimization alone cannot provide
hard geometry or lock guarantees.

### Springer/ECCV: LayoutFlow

Guerreiro et al., *LayoutFlow: Flow Matching for Layout Generation*, ECCV 2024
(Springer Lecture Notes in Computer Science; publisher record to be checked
before publication), motivates a learned continuous layout
ablation. The production decision is intentionally conservative: deterministic
Layout IR remains the baseline because it has a complete renderer and lock
contract, while a learned LayoutFlow branch must first produce the same IR,
source-disjoint split, checkpoint receipt, and constraint-verification metrics.

### Conditional diffusion control

ControlNet's IEEE-adjacent spatial-control result and InstructPix2Pix's
instruction-editing result support separating structural conditioning from
pixel synthesis. Krisna's condition image is therefore a first-class artifact,
not an inferred Canny map or a hidden annotation. This prevents the training
pipeline from claiming a paired-edit objective that the native Qwen trainer
has not yet demonstrated.

### Serving implication

The papers motivate capability boundaries, not a wholesale distributed serving
rewrite. Krisna is a single-GPU local product, so the appropriate production
optimization is admission control, warm baseline residency, asynchronous
background warming, explicit OOM fallback, and low-cost Rust policy checks.
Continuous batching or multi-replica serving should only be added after a
measured concurrent workload demonstrates that one-GPU serialization is the
bottleneck; it is not assumed to improve the interactive design loop.

The current flow-matching upgrade plan is documented in
`docs/research/FLOW_MATCHING_UPGRADE_PLAN.md`. It uses publisher proceedings
and official model/implementation records only; no arXiv-only source is used
for an upgrade decision.

## References

- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image
  Synthesis*, ICML 2024, PMLR 235:
  <https://proceedings.mlr.press/v235/esser24a.html>.
- Brooks et al., *InstructPix2Pix*, CVPR 2023 Open Access:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html>.
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*
  (ControlNet), ICCV 2023 Open Access:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html>.
- Kulikov et al., *FlowEdit: Inversion-Free Text-Based Editing Using
  Pre-Trained Flow Models*, ICCV 2025 Open Access:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html>.
