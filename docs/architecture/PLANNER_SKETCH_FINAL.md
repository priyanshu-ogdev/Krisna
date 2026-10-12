# Planner + Sketch final layered architecture

> **Production decision:** Planner is frozen and llama.cpp-only. Sketch is
> structured Layout IR and wireframe-only. No fallback model is shipped.

## Layer 1 — Product contracts

The frozen Planner produces a validated design-state delta. Sketch produces
`krisna.layout.v1`; neither model may bypass its contract.

## Layer 2 — Planner inference

The consumer release has one Planner runtime: llama.cpp with a pinned GGUF
artifact under `models/planner/`. vLLM and Transformers are rejected release
paths documented in `docs/architecture/LEGACY_REJECTIONS.md`.

Planner training is disabled. Model artifacts belong under `models/planner/`.

## Layer 3 — Sketch selection

Sketch has one release route: structured Layout IR, legality projection, and
deterministic wireframe rendering. LayoutFlow is a future training step within
this same contract, not a second runtime or fallback.

## Layer 4 — Structured layout

The canonical IR uses normalized x1/y1/x2/y2 boxes, typed nodes, parent links,
z-order, text/content fields, lock flags, and explicit relations. Validation
and projection run before rendering. Projection displacement is observable and
must be used for resampling or rejection once a learned generator exists.

## Layer 5 — Rendering and Polish handoff

The deterministic wireframe is an intermediate reference, not the final image.
The IR remains authoritative. Polish receives the wireframe, prompt, active
constraints, and future layout metadata. Handoff quality must be measured for
wireframe leakage before training a layout generator.

## Layer 6 — Training

`training/src/krisna_training/layout/` now contains the canonical JSONL
canonicalizer/dataset and a small continuous layout-flow model primitive. The
config in `training/configs/layout_flow_base.yaml` is a research starting point,
not a validated production recipe. It must be trained only after data licenses,
group splits, taxonomy mappings, and the handoff ablation are complete.

Training entry points for Planner LoRA and Gemma QLoRA have been removed from
the supported configuration surface. Historical Python helpers remain only
because old tests import pure masking/dataset code; no launcher or training
config remains.

## Layer 7 — Evaluation

Evaluate structured legality, box/tree/relation accuracy, lock preservation,
projection displacement, and final rendered UI quality. Layout FID is only
valid with a Krisna-specific feature protocol or an exactly reproduced public
protocol. The current MaskGIT route remains the rollback baseline.

## Research basis

- LayoutFlow, Guerreiro et al., ECCV 2024: https://arxiv.org/abs/2403.18187
- LayoutDiffusion, Zhang et al., ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV2023_paper.pdf
- LayoutDM, Chai et al., CVPR 2023: https://arxiv.org/abs/2305.02567
- UniLayDiff, Liu et al., CVPR 2026: https://arxiv.org/abs/2512.08897
- LayoutCoT: https://arxiv.org/abs/2504.10829
- TA-TiTok/MaskGen, Kim et al., ICCV 2025: https://arxiv.org/abs/2501.07730
- Grammar-constrained decoding, Geng et al., EMNLP 2023: https://aclanthology.org/2023.emnlp-main.674/
- AWQ, Lin et al., MLSys 2024: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf

The complete selection rationale is maintained in `docs/citations.md`.

## Cleanup verdict

The old Planner LoRA launcher is blocked. The Gemma Critic is no longer on the
active critique route. Historical Planner/Critic Python helpers are
compatibility-only for old pure helper tests and must not be described as active
training architecture.
