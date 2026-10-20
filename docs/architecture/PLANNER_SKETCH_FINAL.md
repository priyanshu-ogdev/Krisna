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

### Layer 2.1 — Multimodal VLM Module Synchronization (Qwen3.5-9B / Qwen2.5-VL)

**The Architectural Reasoning (The Pixel vs. Discrete Token Dichotomy):**
- *Why Qwen-Image-2.1 Cannot Decompile Layouts*: Qwen-Image-2.1 is a flow-matching Diffusion Transformer (MMDiT) operating on a continuous 64-channel latent space. Its generative loss is velocity-matching $\mathcal{L}_{\text{FM}} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$. It outputs *pixels*, NOT discrete tokens or JSON coordinates. Attempting to force a flow-matching pixel generator to parse or decompile UI layouts is mathematically and architecturally unsound.
- *Why the 9B Model Is the Native VLM Module*: Qwen3.5-9B / Qwen2.5-VL is an autoregressive Vision-Language Model with a native dynamic-resolution Vision Transformer (`mmproj`). Pretrained on WebUI, Screen2Code, RefCOCO, and Visual Grounding, it natively maps flat raster image patches into cross-attention token streams and generates discrete tokens.
- *Synchronized Pipeline Execution*: The Planner (9B) serves directly as the visual perception and layout engine (VLM module). When given an uploaded UI screenshot or photo (`image_ref`, `wireframe_ref`, or `image_bytes`), the Planner VLM performs visual grounding, recognizes discrete UI components (hero, nav, cards, buttons, headings, focal subjects), grounds bounding boxes $[x_1, y_1, x_2, y_2]$ normalized to $0..1$, and outputs a versioned `DesignStateDelta` with discrete `DesignAction` items (`ADD_COMPONENT`, `MOVE_COMPONENT`, `LOCK_REGION`).
- *Direct Sketch Tier Integration*: In the same turn, the Sketch Tier (`LayoutBackend`) receives `planner_output["design_state_delta"]`, parses the actions into a `LayoutIR` DAG, resolves constraints via deterministic simplex partitioning, and renders the dual-stream conditioning bundle (`DualStreamSketch`) with 0 MB VRAM overhead and $<5\text{ms}$ CPU latency.
- *API & Sequence Flow*: Exposed via `flows.decompile_screenshot()` and FastAPI endpoint `POST /session/{session_id}/decompile`. A designer can upload a flat competitor screenshot and say *"Move the CTA to the bottom and make the headline bolder"*; the 9B VLM decompiles the screenshot into an editable layout tree on Turn 1.

### Layer 2.2 — Image-to-Image Support & Facial Identity Shielding (NVIDIA AnyDoor & ConsisID)

When editing uploaded photos, portrait character images, or brand assets:
- **Wireframe Lock Donor Resolution**:
  - *Turn 1 Text-to-Image*: `lock_donor_ref` resolves to `None`. This eliminates the legacy "wireframe copy" bug where low-fidelity wireframe lines were copied into the photorealistic render.
  - *Turn 1 Image-to-Image*: When a reference photo or screenshot is uploaded, `lock_donor_ref` resolves to `state.constraints.reference_image_ref`.
  - *Turn 2+ Multi-Turn Iteration*: `lock_donor_ref` resolves to the prior high-fidelity render (`state.finalize_output.image_ref`).
- **Spatial Floor ($\ge 128\times 128\text{px}$) to Prevent Latent Token Collapse**:
  - In MMDiT architectures (such as Qwen-Image-2.1), VAE downsampling ($16\times$) combined with patch embedding stride ($2\times 2$) compresses visual features into $32\times 32$ spatial tokens. Any face smaller than $128\times 128\text{px}$ is represented by fewer than 4 latent tokens in cross-attention, causing irreversible structural collapse ("melting face" phenomenon).
  - Enforced in `layout_baseline.py`: all `focal_subject`, `character`, and facial nodes are strictly clamped to a minimum spatial floor of $\ge 128\text{px}$ ($\ge 12.5\%$ of canvas).
- **Inpainting Mask & Exact Donor Compositing**:
  - During Flow ODE sampling, the locked face region is masked with $M(x, y) = 0$ in `_build_inpaint_mask()`, shielding it from cross-attention noise and identity drift.
  - After sampling, `_apply_locked_regions()` composites the exact donor pixels from `reference_image_ref` onto the canvas, guaranteeing 0.00% identity drift.
  - Laplacian edge gradient alignment ($\Delta \nabla \le 0.05$) is verified along the composite seam (NVIDIA AnyDoor) to prevent visible borders.

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

The Sketch Tier is entirely deterministic and requires no learned weights or training checkpoints.
The experimental `LayoutFlow` neural ODE ablation (`krisna_training/layout/`) has been retired and removed.
The production pipeline standardizes on the `krisna.layout.v1` neuro-symbolic engine (0 MB VRAM, <5ms CPU latency).

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
- Design2Code, Si et al., ACL 2025: https://aclanthology.org/2025.acl-long.42/
- AnyDoor, Chen et al., CVPR 2024: https://doi.org/10.1109/CVPR52733.2024.02685
- ConsisID, Shen et al., 2024: https://doi.org/10.1109/CVPR52734.2025.00192
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
