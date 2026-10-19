# Inference Contract: Planner $\to$ Sketch $\to$ Polish Bridge

## 1. End-to-End Pipeline Contract

```text
┌────────────────────────────────────────────────────────┐
│  STAGE 1: PLANNER TIER (Tier.PLANNER)                  │
│  - Input: Session conversational history + user prompt │
│  - Output: DesignStateDelta                            │
│    ├── constraint_updates (style, palette, asset_mode) │
│    ├── actions: [AddComponent, Remove, SetText, Lock]  │
│    └── reasoning_note: Architectural design intent     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: SKETCH TIER (Tier.SKETCH - LayoutBackend)    │
│  - Execution: layout_from_constraints()                │
│    ├── Merges actions into canonical LayoutIR tree     │
│    ├── Synchronizes locked_regions onto LayoutNode     │
│    ├── Flexbox Collision Partitioner (_resolve_flex)   │
│    ├── 8pt Grid Snapping (8 / canvas_dim)              │
│    └── 32px Canvas Invariant (W, H = 0 mod 32)         │
│  - Rasterization: render_wireframe() & render_segmentation_map()│
│    ├── Anti-aliased TrueType fonts (>= 12px)           │
│    ├── Measured pixel text wrapping (_wrap_text)       │
│    ├── Native RGBA transparency when asset_mode=trans  │
│    └── NVIDIA SPADE Canonical Categorical Map (RGB)    │
│  - Output: layout_ref + wireframe_ref + segmentation_ref│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  BRIDGE: DESCRIPTIVE PROMPT SYNTHESIS & LOCK DONOR     │
│  - flows.finalize() / synthesize_polish_prompt()       │
│    ├── Parses LayoutIR nodes into Header/Main/Action   │
│    ├── Enforces Verbatim Label Guard (quoted labels)   │
│    ├── Propagates segmentation_ref to Polish backend   │
│    └── Resolves lock_donor_ref:                        │
│        If prior render exists -> prior render          │
│        Otherwise -> wireframe_ref                      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 3: POLISH TIER (Tier.POLISH - Qwen-Image-2.1)   │
│  - Model: 32-block MMDiT (Joint Single-Stream)         │
│  - Latent: 16x VAE downsampler (f16c64) + 2x2 DiT      │
│  - Conditioning: Multi-Stream Spatial Conditioning:    │
│    ├── Stream 1 (source_img): TrueType Wireframe / Prior│
│    └── Stream 2 (segmentation_img): SPADE Semantic Map │
│  - Synthesis: Flow Matching ODE (Midpoint / Euler)     │
│  - Lock Preservation: Composite exact donor pixels     │
│  - Output: finalize_output.image_ref (Blob)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 4: CLOSED-LOOP VERIFIER STACK                   │
│  - Layout IoU Verifier (Target: >= 0.70)               │
│  - OCR Readability Verifier (Target: >= 0.80)          │
│  - Seam Continuity Verifier (Target: >= 0.85)          │
│  - Safety Gate Verifier (Target: >= 0.90)              │
│  - Automated Closed-Loop Retry:                        │
│    If drift detected -> trigger auto-refine with       │
│    grounded spatial retry prompt referencing layout    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Invariant Specifications

### 2.1 The Wireframe-as-Lock-Donor Invariant
When a user updates an existing design (e.g. Turn 2: "change the button label to 'Submit', keep the hero card unchanged"):
1. The **Planner** marks the hero card as `locked: true` and emits `SetTextAction(target_id="cta", text_content="Submit")`.
2. The **Sketch Tier** renders the updated wireframe. The hero card is pinned in-place, and the CTA button label is updated.
3. The **Bridge** inspects `state.finalize_output.image_ref`. Because a high-resolution render already exists from Turn 1, `lock_donor_ref` is set to Turn 1's photo render, NOT the wireframe!
4. The **Polish Tier** uses the wireframe to guide the newly modified CTA button, but copies the pixels of the hero card directly from the Turn 1 photorealistic render.
5. Result: 100% mathematical pixel preservation on locked regions with zero degradation to wireframe-level quality.

### 2.2 Verbatim Label Guard
Any textual string defined in a `LayoutNode` or `SetTextAction` is tracked as an `expected_label`. 
`synthesize_descriptive_polish_prompt()` runs `assert_and_inject_verbatim_labels()`. If any label is omitted from the synthesized prompt, it is forcibly injected as a quoted label directive:
```text
"... The header displays 'AI Analytics Dashboard'. CRITICAL TEXT REQUIREMENT: Include exact verbatim text 'AI Analytics Dashboard' in the header element."
```

### 2.3 Closed-Loop Auto-Refine Feedback Loop
If the Verifier Stack detects spatial or typographic drift after Polish generation:
- `layout_iou < 0.70`,
- `ocr_readability < 0.80`, or
- `seam_continuity < 0.85`

The pipeline automatically constructs a grounded spatial retry prompt via `build_grounded_spatial_retry_prompt()` referencing the exact component types and bounding boxes from the `LayoutIR`, executing an automated retry turn. If the retry improves IoU, the improved render is accepted; otherwise, the diagnostic is recorded in the session history for the next Planner turn.

### 2.4 NVIDIA SPADE-Grounded Dual-Stream Spatial Conditioning Invariant
Passing only a rasterized text wireframe to a flow-matching diffusion transformer introduces ambiguity: subtle component boundaries (such as a card container vs. an interactive input field) can suffer from feature normalization washout (Park et al., CVPR 2019 Oral / IEEE TPAMI 2021).

To guarantee unambiguous categorical regional conditioning:
1. The **Sketch Tier** invokes `render_segmentation_map()` in tandem with `render_wireframe()`.
2. Every UI node in `LayoutIR` is rasterized to an exact 3-channel RGB categorical mask using the canonical `UI_SEMANTIC_PALETTE`:
   - `background`: `(240, 240, 240)` (0)
   - `container`: `(180, 180, 180)` (1)
   - `card`: `(200, 200, 220)` (2)
   - `text`: `(255, 200, 0)` (3)
   - `button`: `(0, 120, 255)` (4)
   - `input`: `(0, 200, 120)` (5)
   - `image`: `(255, 100, 100)` (6)
   - `icon`: `(200, 100, 255)` (7)
   - `divider`: `(120, 120, 120)` (8)
   - `list`: `(100, 220, 220)` (9)
   - `other`: `(150, 150, 150)` (10)
   - `locked`: `(255, 0, 128)` (11)
3. Both `wireframe_ref` and `segmentation_ref` are persisted to `BlobStore` and stored in `DesignState.layout`.
4. In `flows.finalize()`, `polish_input` receives both `image_ref` and `segmentation_ref`.
5. `QwenImage21Backend` resolves `segmentation_ref` and supplies `[source_img, segmentation_img]` as multi-stream condition images into the MMDiT attention pipeline, grounding cross-attention maps and eliminating categorical hallucination.
6. Execution telemetry logs `segmentation_used: true` and `spatial_conditioning_mode: "NVIDIA_SPADE_MultiStream"`.
