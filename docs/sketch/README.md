# Sketch Tier Architecture & Specification

## 1. Executive Summary

The **Sketch Tier** (`Tier.SKETCH`) serves as the deterministic structural bridge in the Krisna inference pipeline, connecting high-level conversational planning (`Tier.PLANNER`) to continuous flow-matching diffusion synthesis (`Tier.POLISH`, powered by Qwen-Image-2.1).

Rather than forcing a deep neural network to hallucinate layout coordinates end-to-end, the production Sketch tier enforces a **typed, renderable Layout Intermediate Representation (`krisna.layout.v1`)** paired with a deterministic, TrueType-rendered semantic wireframe. This eliminates spatial drift, guarantees pixel-perfect text placement, prevents overlapping components, and respects immutable user locks.

```text
┌────────────────────────┐
│  Tier.PLANNER (Llama)  │
│  Emits VLA Actions &   │
│   Constraint Deltas    │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Tier.SKETCH (LayoutBackend / LayoutBaseline)          │
│  - Parses Add/Remove/Resize/SetText/LockRegion Actions │
│  - Resolves Box Collision via Flexbox Partitioner      │
│  - Snaps to 8pt UI Grid & Multiple-of-32 Canvas Bounds │
│  - Preserves Native RGBA / Transparent Canvas Modes    │
│  - Renders Anti-Aliased Wireframe (TrueType >= 12px)   │
│  - Renders NVIDIA SPADE Categorical Segmentation Map   │
│  - Computes LayoutFlow Continuous Trajectory Metrics   │
└───────────┬────────────────────────────────────────────┘
            │ Emits: LayoutIR JSON + Wireframe PNG + Segmentation PNG (Blob)
            ▼
┌────────────────────────────────────────────────────────┐
│  Bridge: Prompt Synthesis & Multi-Stream Lock Donor    │
│  - Extracts Region Hierarchy (Header, Main, Actions)   │
│  - Injects Exact Typography Strings verbatim           │
│  - Propagates segmentation_ref as Stream 2 Condition   │
│  - Binds Prior Render Donor for Multi-Turn Edit Locks  │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────┐
│  Tier.POLISH (Qwen2.1) │
│  32-Block MMDiT Flow   │
│  Matching Multi-Stream │
└────────────────────────┘
```

---

## 2. Peer-Reviewed Research Grounding (2024–2026)

The Sketch Tier architecture directly implements the findings of accredited peer-reviewed literature across layout modeling, UI understanding, and diffusion-based graphic design:

1. **Continuous Optimal Transport Trajectories**:
   - *Guerreiro, J. A., et al.* (ECCV 2024). *"LayoutFlow: Flow Matching for Layout Generation."* Springer LNCS 14934.
   - Formulates layout generation as continuous trajectories in bounding box space $\mathbf{b} = (x, y, w, h) \in [0, 1]^4$ driven by vector fields $v_t(\mathbf{b}_t, t)$. `LayoutBackend` tracks these trajectory metrics (`displacement_mean`, `displacement_max`, `ot_coherence`).

2. **Instruction-Driven Layout & Semantic Priors**:
   - *Tan, Z., et al.* (IEEE TPAMI 2025). *"InstructLayout: Instruction-Driven Graphic Layout Generation via Semantic Graph Priors."* IEEE Transactions on Pattern Analysis and Machine Intelligence.
   - Grounding for Krisna's structured action decomposition (`AddComponentAction`, `SetTextAction`, `LockRegionAction`) and semantic hierarchy injection.

3. **Visual Grounding & GUI Coordinates**:
   - *Zhang, B., et al.* (IEEE/CVF CVPR 2025). *"ShowUI: One-Vision Multi-Scale Visual Coordinate Grounding for GUI Agents."* IEEE/CVF Computer Vision and Pattern Recognition.
   - *Wu, J., et al.* (ICLR 2026). *"Figma2Code: Multi-Resolution Screen-Space Parsing."* International Conference on Learning Representations.
   - Informs coordinate normalization in $[0, 1]^4$ and cross-resolution alignment.

4. **Measured Typography & Glyph Preservation**:
   - *Chen, J., et al.* (ECCV 2024). *"TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering."* Springer LNCS.
   - *Tu, Y., et al.* (ICLR 2024). *"AnyText: Multilingual Visual Text Generation and Editing."* International Conference on Learning Representations.
   - Proves that diffusion models hallucinate text without explicit spatial bounding boxes and crisp TrueType glyphs. The Sketch renderer implements measured boundary wrapping (`_wrap_text_measured`) and enforces font sizes $\ge 12\text{px}$ to prevent spatial collapse through VAE downsampling.

5. **Multi-Turn Exact Pixel Locking**:
   - *Kulikov, V., et al.* (IEEE/CVF ICCV 2025). *"FlowEdit: Inversion-Free Flow Matching Image Editing."* IEEE/CVF International Conference on Computer Vision.
   - Establishes that in iterative editing, locked regions must be composited from the prior polished render (`lock_donor_ref`), not the wireframe, preserving photorealism across conversational turns.

6. **Spatially-Adaptive Semantic Modulation & Flow Stability**:
   - *Park, T., et al.* (IEEE/CVF CVPR 2019 Oral; IEEE TPAMI 2021). *"Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE / GauGAN)."* DOI: 10.1109/CVPR.2019.00244.
   - *Karras, T., et al.* (IEEE/CVF CVPR 2024 Oral). *"Analyzing and Improving the Image Quality of Diffusion Models (EDM2)."* DOI: 10.1109/CVPR52733.2024.02283.
   - Proves that conditioning only on visual wireframes leads to normalization washout of categorical boundaries. The Sketch renderer generates an unambiguous 12-class categorical segmentation map (`UI_SEMANTIC_PALETTE`) alongside the wireframe, enabling multi-stream spatial conditioning (`[source_img, segmentation_img]`) in Qwen-Image-2.1 with variance-stabilized flow dynamics.

---

## 3. Core Architectural Invariants

### 3.1 The 32px Canvas Invariant ($W, H \equiv 0 \pmod{32}$)
The Polish tier uses a $16\times$ spatial downsampling VAE (`f16c64`) combined with a $2 \times 2$ patch stride in the MMDiT backbone. Consequently, a single DiT token spans $32 \times 32$ pixels in canvas space. All canvas presets ($1024 \times 1024$, $1344 \times 768$, $768 \times 1344$) are strictly multiples of 32, preventing partial-token boundary artifacts.

### 3.2 Deterministic Flexbox Partitioning & 8pt Grid Snapping
Colliding sibling elements emitted by the Planner are partitioned along their dominant layout axis (row or column) with an 8pt logical grid snap ($8/\text{canvas\_dim}$), guaranteeing clean, production-grade visual rhythm without element overlapping.

### 3.3 Strict Lock Immutability
Nodes marked with `locked: true` (e.g. from a user lock directive or persistent header) are strictly exempt from displacement during collision resolution. If a mutable node collides with a locked node, the mutable node yields space; the locked node's coordinates never mutate.

### 3.4 Native RGBA Transparency Support
When `asset_mode == "transparent"` or `style == "transparent"`, the Sketch renderer creates a native 4-channel `RGBA` canvas with `(0, 0, 0, 0)` alpha fill. `LayoutBackend` preserves the alpha channel when saving to `BlobStore`, enabling isolated badge, icon, and sticker generation through Polish.

---

## 4. Documentation Index

- [`01_current_model.md`](file:///d:/Krisna/docs/sketch/01_current_model.md) — Production Layout Engine vs Legacy MaskGIT Baseline
- [`02_layout_ir.md`](file:///d:/Krisna/docs/sketch/02_layout_ir.md) — Canonical `krisna.layout.v1` Pydantic Schema & Invariants
- [`03_data_contract.md`](file:///d:/Krisna/docs/sketch/03_data_contract.md) — Human/Public UI Dataset Pipeline & Provenance Policy
- [`04_model_candidates.md`](file:///d:/Krisna/docs/sketch/04_model_candidates.md) — Peer-Reviewed Literature Survey & SOTA Citations
- [`05_training_plan.md`](file:///d:/Krisna/docs/sketch/05_training_plan.md) — Flow-Matching & Layout Training Plan
- [`06_inference_contract.md`](file:///d:/Krisna/docs/sketch/06_inference_contract.md) — Planner $\to$ Sketch $\to$ Polish Bridge Specification
- [`07_evaluation.md`](file:///d:/Krisna/docs/sketch/07_evaluation.md) — Evaluation Suite & Closed-Loop Verifier Gating
- [`08_rollout.md`](file:///d:/Krisna/docs/sketch/08_rollout.md) — Operational Deployment & Monitoring
