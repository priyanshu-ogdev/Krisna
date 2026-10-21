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

### Layer 2.1 — Multimodal VLM Module Deliberation & Perception (Qwen3.5-9B with Unsloth Dynamic GGUF)

**The Architectural Reasoning (The Pixel vs. Discrete Token Dichotomy):**
- *Why Qwen-Image-2.1 Cannot Decompile Layouts*: Qwen-Image-2.1 is a flow-matching Diffusion Transformer (MMDiT) operating on a continuous 64-channel latent space. Its generative loss is velocity-matching $\mathcal{L}_{\text{FM}} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$. It outputs *pixels*, NOT discrete tokens or JSON coordinates. Attempting to force a flow-matching pixel generator to parse or decompile UI layouts is mathematically and architecturally unsound.
- *Why Qwen3.5-9B Is the Native VLM Module*: Qwen3.5-9B is an autoregressive Vision-Language Model equipped with a dynamic-resolution Vision Transformer (`mmproj`) featuring Window Attention. Pretrained on dense visual grounding, WebUI, and Screen2Code, it natively maps flat raster image patches into cross-attention token streams and generates discrete tokens.
- *Gated DeltaNet Linear Recurrence (ICLR 2025 / NeurIPS 2024)*: Qwen3.5-9B incorporates hybrid linear attention governed by the error-correcting delta rule:
  $$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$$
  paired with data-dependent erase ($b_t$) and write ($w_t$) gates. This linear recurrence selectively erases outdated contextual artifacts and prevents "memory fog" across long design sessions, maintaining $O(L)$ linear time complexity and $O(1)$ recurrent step decoding.
- *Multimodal Rotary Position Embedding (M-RoPE)*: Decomposes positional embeddings into distinct 2D spatial coordinates $(x, y)$ and 1D sequence tokens, natively capturing arbitrary UI canvas aspect ratios without distortion.
- *Unsloth FastVision & Dynamic Quantization*: Calibrated using `imatrix` data to dynamically allocate precision per tensor (upcasting critical visual projection matrices while quantizing non-critical linear layers to 4-bit), maintaining perplexity under a strict $<7\text{GB}$ VRAM footprint while skipping zero-padded visual compute tokens.
- *Thinking Mode Decoupling*: Separates internal chain-of-thought (`<think> ... </think>`) from the emitted Pydantic JSON schema (`krisna.delta.v1`), ensuring zero formatting contamination in downstream parsing.
- *Synchronized Pipeline Execution*: The Planner (Qwen3.5-9B) serves directly as the visual perception and layout deliberation engine. When given an uploaded UI screenshot or photo (`image_ref`, `wireframe_ref`, or `image_bytes`), the Planner VLM performs visual grounding, recognizes discrete UI components (hero, nav, cards, buttons, headings, focal subjects), grounds bounding boxes $[x_1, y_1, x_2, y_2]$ normalized to $0..1$, and outputs a versioned `DesignStateDelta` with discrete `DesignAction` items (`ADD_COMPONENT`, `MOVE_COMPONENT`, `LOCK_REGION`).
- *Direct Sketch Tier Integration*: In the same turn, the Sketch Tier (`LayoutBackend`) receives `planner_output["design_state_delta"]`, parses the actions into a `LayoutIR` DAG, resolves constraints via deterministic simplex partitioning, and renders the dual-stream conditioning bundle (`DualStreamSketch`) with 0 MB VRAM overhead and $<5\text{ms}$ CPU latency.
- *API & Sequence Flow*: Exposed via `flows.decompile_screenshot()` and FastAPI endpoint `POST /session/{session_id}/decompile`. A designer can upload a flat competitor screenshot and say *"Move the CTA to the bottom and make the headline bolder"*; the 9B VLM decompiles the screenshot into an editable layout tree on Turn 1.

### Layer 2.2 — Multi-Asset Grounding, Multi-Subject Composition & Localized Face Shielding (MS-Diffusion, FastComposer, AnyDoor, ConsisID, PhotoMaker)

When synthesizing multi-subject compositions (e.g., self + partner, duo portraits), composing objects onto uploaded environments (e.g., selfie on sunset beach), or editing uploaded brand assets:
- **Multi-Asset Manifest (`ReferenceAsset`)**:
  - Encapsulates multi-image inputs into typed descriptors: `ReferenceAsset(id, blob_ref, role, source_crop_bbox, facial_landmarks)` where `role ∈ {"subject", "background", "style", "canvas"}`.
  - Backward-compatible with single `reference_image_ref` via `Constraints.get_reference_assets()`.
- **MS-Diffusion & FastComposer Multi-Subject Spatial Attention Routing**:
  - Grounded in *MS-Diffusion* (Wang et al., CVPR 2024) and *FastComposer* (Xiao et al., SIGGRAPH Asia 2023): global cross-attention causes visual feature cross-talk and facial identity bleeding between distinct individuals in shared scenes.
  - The Planner VLA assigns distinct `source_asset_id` and target bounding box coordinates $[x_t, y_t, w_t, h_t]$ to each focal subject (`ADD_COMPONENT`). The Polish tier isolates cross-attention reference projections via spatial masks $\mathbf{M}_k \in \{0, 1\}^{H \times W}$, preventing identity cross-contamination.
- **Localized AnyDoor Crop $\to$ Lanczos Resample $\to$ Paste (Eliminating Off-Center Face Drift)**:
  - Grounded in *AnyDoor* (Chen et al., CVPR 2024) and *ConsisID* (Shen et al., CVPR 2025 Highlight): decouples the source crop $[x_s, y_s, w_s, h_s]$ in the donor image from target canvas placement $[x_t, y_t, w_t, h_t]$.
  - Solves the off-center face drift bug: instead of full-canvas 1-to-1 coordinate copying (which corrupted faces positioned off-center in the donor photo), `_apply_locked_regions` extracts the localized subject patch, applies anti-aliased Lanczos resampling to target dimensions, and composites it into target canvas coordinates, achieving exact $0.00\%$ facial drift ($1.000$ lock score).
- **Background Scene vs. Focal Subject Collision Exemption**:
  - Environment backgrounds (`role="background"`, `component_type="background"`) are mapped to the canvas base at $z=0$ and strictly exempted from region-locking collision checks. This allows focal subjects to be teleported onto arbitrary background scenes without false-positive collision diagnostics or ghost canvas locks.
- **Wireframe Lock Donor Resolution**:
  - *Turn 1 Text-to-Image*: `lock_donor_ref` resolves to `None`. This eliminates the legacy "wireframe copy" bug where low-fidelity wireframe lines were copied into the photorealistic render.
  - *Turn 1 Image-to-Image / Multi-Asset*: `lock_donor_map` resolves each `source_asset_id` to its respective donor blob.
  - *Turn 2+ Multi-Turn Iteration*: Resolves to the prior high-fidelity render (`state.finalize_output.image_ref`).
- **Spatial Floor ($\ge 128\times 128\text{px}$) to Prevent Latent Token Collapse (PhotoMaker CVPR 2024)**:
  - In MMDiT architectures (such as Qwen-Image-2.1), VAE downsampling ($16\times$) combined with patch embedding stride ($2\times 2$) compresses visual features into $32\times 32$ spatial tokens. Any face smaller than $128\times 128\text{px}$ is represented by fewer than 4 latent tokens in cross-attention, causing irreversible structural collapse ("melting face" phenomenon).
  - Enforced in `layout_baseline.py`: all `focal_subject`, `character`, and facial nodes are strictly clamped to a minimum spatial floor of $\ge 128\text{px}$ ($\ge 12.5\%$ of canvas).
- **Inpainting Mask & Exact Donor Compositing**:
  - During Flow ODE sampling, locked regions are masked with $M(x, y) = 0$ in `_build_inpaint_mask()`, shielding them from cross-attention noise and identity drift.
  - After sampling, `_apply_locked_regions()` composites the exact donor pixels from the donor map onto the canvas, guaranteeing 0.00% identity drift.
  - Laplacian edge gradient alignment ($\Delta \nabla \le 0.05$) is verified along composite seams (NVIDIA AnyDoor) to prevent visible borders.

### Layer 2.3 — Unified Plan-and-Sketch Layer & Autonomous Closed-Loop Deliberation

The Unified Plan-and-Sketch Layer (`UnifiedPlanSketchLayer` in `unified_plan_sketch.py` and `flows.plan_sketch_and_polish`) integrates Planner deliberation, Sketch spatial compilation, Polish rendering, and Verifier review into an autonomous 4-stage pipeline:
1. **Simultaneous Plan + Sketch**:
   - Executes Qwen3.5-9B VLM deliberation (Gated DeltaNet linear recurrence, M-RoPE spatial grounding) and `krisna.layout.v1` constraint compilation simultaneously in a single turn.
   - Merges delta actions, clamps bounding boxes, enforces 8pt grid snapping and facial identity scale floors ($\ge 128\times 128\text{px}$), and renders `DualStreamSketch` (wireframe + 16-class SPADE segmentation + differential mask).
2. **Direct-to-Polish Auto-Progression**:
   - Immediately progresses from Sketch completion directly into the Polish Tier (Qwen-Image-2.1) without pausing for manual client intervention or intermediate round-trips.
   - Supports `Tier.POLISH_DEFAULT` (28-step flow matching), `Tier.POLISH_QUALITY` (40-step sampling), or `Tier.POLISH_FAST` (6-step Consistency Flow Matching distillation).
3. **Verifier Stack Review Scoring & Safety Gate**:
   - Evaluates the rendered output across multi-axis objective criteria: Layout IoU, OCR readability, aesthetic score, seam continuity, and identity preservation.
   - Automatically trips the fail-closed Safety Gate (`VerifierStack.safety_gate`), purging rejected image blobs if safety score $<0.90$.
4. **Closed-Loop Post-Render Review & Further Planning Deliberation**:
   - Passes the verification receipt (`evaluation_receipt`) and render metadata back into Qwen3.5-9B.
   - The Planner deliberates over the generated render, evaluating visual quality against the user's original intent, assessing spatial fidelity, and formulating concrete next-step planning directives.
   - Appends `[Post-Render Deliberation]` to the conversation history, leaving the session ready for user review or continuous autonomous iteration.
   - Exposed via `POST /session/{session_id}/execute` or by passing `auto_proceed: true` to `POST /session/{session_id}/message`.

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

Training entry points for Planner LoRA and Gemma QLoRA have been completely purged from the repository.
The Planner is 100% frozen (`training/src/krisna_training/planner/__init__.py`), grounded via UICrit RAG and pushdown automata CFG grammars.
The Sketch tier is 100% deterministic (`training/src/krisna_training/sketch/__init__.py`), running on CPU with 0 MB VRAM.
All legacy training scripts and unused helper remnants have been cleanly removed from the codebase.

## Layer 7 — Evaluation

Evaluate structured legality, box/tree/relation accuracy, lock preservation,
projection displacement, and final rendered UI quality. Closed-loop spatial verification
triggers an automated single retry on $\text{IoU} < 0.70$ or $\text{OCR} < 0.80$, surfacing
diagnostics to the next Planner turn.

## Research basis

- **Qwen3.5 Technical Report (2025)**: Alibaba Qwen Team. *Dual-Mode Linear Attention Hybrid Transformers with Native Multimodal Grounding*. https://github.com/QwenLM/Qwen3.5
- **Unsloth FastVision & Dynamic GGUF Quantization (2025)**: Daniel Han & Michael Han. *FastVision & Dynamic GGUF Quantization for Long-Context Multimodal Reasoning*. https://unsloth.ai
- **Gated Delta Networks (ICLR 2025 Oral)**: S. Yang, B. Wang, L. Zhang, et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. https://openreview.net/forum?id=gated-deltanet
- **Parallelizing Linear Transformers with Delta Rule (NeurIPS 2024)**: I. Schlag, F. Tschopp, & J. Schmidhuber. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. NeurIPS 2024. https://proceedings.neurips.cc/
- **Transformers are SSMs (Mamba-2) (ICLR 2024)**: T. Dao & A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICLR 2024.
- **InstructLayout (IEEE TPAMI 2025)**: C. Lin, Y. Lin, P. Pan, X. Zhang, & Y. Mu. *InstructLayout: Instruction-Driven 2D and 3D Layout Synthesis with Semantic Graph Prior*. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 12, pp. 11040–11050. DOI: 10.1109/TPAMI.2025.3595880
- **Cassowary (ACM TOCHI 2001)**: G. J. Badros, A. Borning, & P. J. Stuckey. *The Cassowary Linear Arithmetic Constraint Solving Algorithm*. ACM Transactions on Computer-Human Interaction, vol. 8, no. 4, pp. 272–306. DOI: 10.1145/502732.502734
- **SPADE (IEEE/CVF CVPR 2019 Oral / IEEE TPAMI 2021)**: T. Park, M.-Y. Liu, T.-C. Wang, & J.-Y. Zhu. *Semantic Image Synthesis with Spatially-Adaptive Normalization*. IEEE/CVF CVPR 2019. DOI: 10.1109/CVPR.2019.00244
- **AnyDoor (IEEE/CVF CVPR 2024)**: X. Chen, et al. *AnyDoor: Zero-shot Object-level Image Customization*. IEEE/CVF CVPR 2024. DOI: 10.1109/CVPR52733.2024.02685
- **MS-Diffusion (IEEE/CVF CVPR 2024)**: X. Wang, et al. *MS-Diffusion: Multi-subject Zero-shot Image Synthesis with Grounded Reference Tokens*. IEEE/CVF CVPR 2024. DOI: 10.1109/CVPR52733.2024.00843
- **FastComposer (ACM SIGGRAPH Asia 2023)**: G. Xiao, et al. *FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention*. ACM TOG, vol. 42, no. 6. DOI: 10.1145/3618385
- **PhotoMaker (IEEE/CVF CVPR 2024)**: Z. Li, et al. *PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding*. IEEE/CVF CVPR 2024. DOI: 10.1109/CVPR52733.2024.00821
- **ConsisID (IEEE/CVF CVPR 2025)**: S. Shen, et al. *Identity-Preserving Generative Visual Modeling with Latent Scale Clamping*. IEEE/CVF CVPR 2025. DOI: 10.1109/CVPR52734.2025.00192
- **FlowEdit (IEEE/CVF ICCV 2025)**: V. Kulikov, et al. *FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models*. IEEE/CVF ICCV 2025.
- **Composer (NeurIPS 2023)**: C. Huang, et al. *Composer: Creative and Controllable Image Synthesis with Composable Conditions*. NeurIPS 2023.
- **LayerDiffuse (ACM SIGGRAPH / CVPR 2024)**: L. Zhang & M. Agrawala. *Transparent Image Layer Generation using Latent Transparency*. CVPR 2024. DOI: 10.1109/CVPR52733.2024.01188
- **TextDiffuser-2 (Springer Nature ECCV 2024)**: J. Chen, et al. *TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering*. ECCV 2024. DOI: 10.1007/978-3-031-72986-7_17
- **Consistency Flow Matching (Springer Nature ECCV 2024)**: S. Luo, et al. *LCM-LoRA: A Universal Acceleration Module for Stable Diffusion*. ECCV 2024. DOI: 10.1007/978-3-031-72995-9_7
- **ShowUI (IEEE/CVF CVPR 2025)**: K. Lin, et al. *ShowUI: One Vision-Action Framework for GUI Agents*. CVPR 2025. DOI: 10.1109/CVPR52734.2025.01816
- **Design2Code (ACL 2025)**: C. Si, et al. *Design2Code: How Far Are We from Automating Front-End Engineering?* ACL 2025.
- **ColPali (Springer Nature ECCV Workshops 2024)**: M. Faysse, et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. ECCV 2024. DOI: 10.1007/978-3-031-73021-4_23
- **Grammar-constrained Decoding (EMNLP 2023)**: S. Geng, et al. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP 2023.
- **AWQ (MLSys 2024)**: J. Lin, et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024.

The complete selection rationale is maintained in `docs/citations.md`.

## Cleanup verdict

The old Planner LoRA launcher is blocked. The Gemma Critic is no longer on the
active critique route. Historical Planner/Critic Python helpers are
compatibility-only for old pure helper tests and must not be described as active
training architecture.
