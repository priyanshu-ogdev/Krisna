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

Planner deliberation is grounded via:
- **InstructLayout (IEEE TPAMI 2025)**: Structured semantic graph prefix injection
  (`[Structured Layout IR Graph]`) formatting resolved node bounds, element types,
  and lock states into the system prompt.
- **Chat Template Safety**: Interleaved `system` turns from verifier drift diagnostics
  are mapped to `role="user"` prompts with `"[System Diagnostic]: ..."` prefixes.
- **ColPali (ECCV 2024) / UICrit RAG**: Multi-vector visual retrieval directly from
  2D UI screenshot embeddings using pre-cached L2 norms.

Planner training is disabled. Model artifacts belong under `models/planner/`.

## Layer 3 — Sketch selection

Sketch has one release route: structured Layout IR, deterministic Flexbox collision
resolution with locked node immunity, legality projection, and deterministic high-entropy
wireframe rendering. LayoutFlow is a future training step within this same contract,
not a second runtime or fallback.

## Layer 4 — Structured layout

The canonical IR uses normalized x1/y1/x2/y2 boxes, typed nodes, parent links,
z-order, text/content fields, lock flags, and explicit relations:
- **Deterministic Flexbox Partitioning**: Colliding mutable sibling clusters ($\text{IoU} > 0.05$)
  are cleanly partitioned along primary layout axes while strictly exempting locked nodes
  (`node.locked == True`) from any movement or resizing.
- **8pt Grid Snapping & Dynamic Canvas**: Canvas dimensions strictly pinned to multiples
  of 32 ($1024\times 1024$ square, $1344\times 768$ desktop 16:9, $768\times 1344$ mobile 9:16, $1152\times 864$ tablet 4:3)
  meeting Qwen3-VL vision encoder $16\times 2 = 32$ visual patch token grouping and VAE $f=16$ latent compression.
- Validation and projection run before rendering. Projection displacement is observable and
  flow metrics (displacement mean, trajectory steps, vector field norm) are computed.

## Layer 5 — Rendering and Polish handoff

The deterministic wireframe is rendered as a high-entropy 3-channel semantic RGB image:
- **Typography & Elevation**: Anti-aliased TrueType font scaling across Title (22px), Header (18px),
  Body (14px), Button (14px), and Badge (11px) archetypes with luminance elevation drop shadows
  (TextDiffuser-2, AnyText).
- **Handoff Contract**: Polish receives the wireframe in standard 3-channel RGB.
- **Wireframe-as-Lock-Donor Resolution**: When refining or editing an existing render, locked
  regions composite pixels from the *prior photorealistic render* (`lock_donor_ref`), NOT the
  wireframe, guaranteeing exact 1.000 lock preservation (0.00% pixel drift).

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
projection displacement, and final rendered UI quality. Closed-loop spatial verification
triggers an automated single retry on $\text{IoU} < 0.70$ or $\text{OCR} < 0.80$, surfacing
diagnostics to the next Planner turn.

## Research basis

- LayoutFlow, Guerreiro et al., ECCV 2024: https://doi.org/10.1007/978-3-031-72764-1_4
- TextDiffuser-2, Chen et al., ECCV 2024: https://doi.org/10.1007/978-3-031-72986-7_17
- AnyText, Tu et al., ICLR 2024: https://openreview.net/forum?id=unwtwz6r8p
- Cassowary, Badros et al., ACM TOCHI 2001: https://doi.org/10.1145/502732.502734
- FlowEdit, Kulikov et al., ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html
- Scaling Rectified Flow Transformers, Esser et al., ICML 2024: https://proceedings.mlr.press/v235/esser24a.html
- InstructLayout, IEEE TPAMI 2025: https://doi.org/10.1109/TPAMI.2025.3595880
- ShowUI, Lin et al., CVPR 2025: https://doi.org/10.1109/CVPR52734.2025.01816
- ColPali, Faysse et al., ECCV 2024: https://doi.org/10.1007/978-3-031-72995-9_24
- Grammar-constrained decoding, Geng et al., EMNLP 2023: https://aclanthology.org/2023.emnlp-main.674/
- AWQ, Lin et al., MLSys 2024: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf

The complete selection rationale is maintained in `docs/citations.md`.

## Cleanup verdict

The old Planner LoRA launcher is blocked. The Gemma Critic is no longer on the
active critique route. Historical Planner/Critic Python helpers are
compatibility-only for old pure helper tests and must not be described as active
training architecture.
