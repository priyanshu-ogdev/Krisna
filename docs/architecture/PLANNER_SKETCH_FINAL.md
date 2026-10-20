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

Sketch has one authoritative release route: an open-domain **Universal Neuro-Symbolic Canvas Engine**
handling infinite design scenarios (UI mockups, marketing posters, editorial publishing, e-commerce showcases,
infographics, and freeform sketches). It pairs structured Layout IR with deterministic collision resolution,
locked node immunity, legality projection, and deterministic high-entropy dual-stream spatial rendering.
LayoutFlow continuous flow matching is an aligned training step within this same contract.

## Layer 4 — Structured layout (Universal Scene DAG)

The canonical IR uses normalized x1/y1/x2/y2 boxes, polymorphic typed nodes, parent links,
z-order, text/content fields, lock flags, and explicit relations:
- **Universal Polymorphic Node Schema (`NODE_TYPES`)**:
  - *Containers*: `container`, `card`, `region`, `canvas_zone`, `modal`, `background`
  - *Typography*: `text`, `headline`, `subhead`, `body_text`, `caption`, `quote`, `label`
  - *Media & Subjects*: `image`, `focal_subject`, `illustration`, `character`, `product`, `photo_frame`
  - *Functional Elements*: `button`, `input`, `list`, `table`, `navbar`, `footer`, `sidebar`
  - *Graphic Design & Brand*: `badge`, `icon`, `logo`, `divider`, `shape`, `banner`, `decorative`
  - *Freeform & Annotations*: `sketch`, `contour`, `overlay`, `other`
- **Deterministic Simplex Partitioning**: Colliding mutable sibling clusters ($\text{IoU} > 0.05$)
  are cleanly partitioned along primary layout axes while strictly exempting locked nodes
  (`node.locked == True`) from any movement or resizing.
- **8pt Grid Snapping & Dynamic Canvas**: Canvas dimensions strictly pinned to multiples
  of 32 ($1024\times 1024$ square, $1344\times 768$ desktop 16:9, $768\times 1344$ mobile 9:16, $1152\times 864$ tablet 4:3)
  meeting Qwen vision encoder $16\times 2 = 32$ visual patch token grouping and VAE $f=16$ latent compression.
- **Automated Differential Masking ($M_\Delta$)**: Computes turn-to-turn graph diffs, isolating
  modified partitions into an 8-bit mask with a 16px dilation radius, keeping untouched partitions at $0$.
- Validation and projection run before rendering. Projection displacement is observable and
  flow metrics (displacement mean, trajectory steps, vector field norm) are computed.

## Layer 5 — Rendering and Polish handoff

The Sketch Tier produces an atomic `DualStreamSketch` conditioning bundle (NVIDIA Composer, SPADE, LayerDiffuse):
- **Stream 1: High-Frequency TrueType Wireframe (RGB / RGBA)**:
  - Anti-aliased TrueType font scaling across Display (30px), Title (22px), Header (18px),
    Body (14px), Button (14px), Caption (12px), and Badge (12px) archetypes with a strict legibility
    floor ($\ge 12\text{px}$) to prevent spatial collapse across the $16\times$ VAE downsampler (TextDiffuser-2, AnyText).
  - Elevation-aware multi-layer drop shadows scaling with $z$-index ($s = 2+2z, r = 8+2z$).
  - Native 4-channel RGBA transparency when `asset_mode == "transparent"`.
- **Stream 2: Universal 16-Class SPADE Semantic Map (RGB)**:
  - Discrete RGB class palette (`UNIVERSAL_SEMANTIC_PALETTE`) mapping visual and design roles
    (`container`, `card`, `text`, `headline`, `button`, `input`, `focal_subject`, `badge`, `banner`,
    `shape`, `modal`, `sketch`, `divider`, `list`, `locked`) to prevent feature cancellation in deep MMDiT layers.
- **Stream 3: Spatial Alpha / Inpaint Mask (8-bit Grayscale L)**:
  - Binary/soft mask with $0$ on invariant donor regions and $255$ on active edit zones,
    guiding localized FlowEdit trajectory integration.
- **2D Spatial Quad-Tree Prompt Indexing**:
  - Automatically indexes multi-column horizontal partitions (side-by-side cards, columns, split grids)
    with normalized bounding-box coordinates `[x1, y1, x2, y2]` and verbatim quoted labels in the prompt prose.
- **Handoff Contract**:
  - Polish receives multi-stream spatial conditions: `condition_images = [wireframe_img, segmentation_img]`.
  - Verbatim text strings are injected into prompt prose (`assert_and_inject_verbatim_labels()`).
  - When editing or refining an existing render, locked regions composite pixels from the *prior photorealistic render* (`lock_donor_ref`), NOT the wireframe, guaranteeing exact 1.000 lock preservation (0.00% pixel drift).
  - Seam boundary continuity is verified along composited edges via Laplacian gradient ($\Delta \nabla \le 0.05$) adhering to NVIDIA AnyDoor.

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
- ColPali, Faysse et al., ECCV Workshops 2024: https://doi.org/10.1007/978-3-031-73021-4_23
- Hybrid Layout Control for DiT, Wu et al., ICCV 2025: https://openaccess.thecvf.com/
- Composer Multi-Stream Controllable Generation, Huang et al., NeurIPS 2023: https://proceedings.neurips.cc/
- SPADE Semantic Image Synthesis, Park et al., CVPR 2019 Oral / TPAMI 2021: https://doi.org/10.1109/CVPR.2019.00244
- LayerDiffuse, Zhang & Agrawala, CVPR 2024: https://doi.org/10.1109/CVPR52733.2024.01188
- Grammar-constrained decoding, Geng et al., EMNLP 2023: https://aclanthology.org/2023.emnlp-main.674/
- AWQ, Lin et al., MLSys 2024: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf

The complete selection rationale is maintained in `docs/citations.md`.

## Cleanup verdict

The old Planner LoRA launcher is blocked. The Gemma Critic is no longer on the
active critique route. Historical Planner/Critic Python helpers are
compatibility-only for old pure helper tests and must not be described as active
training architecture.
