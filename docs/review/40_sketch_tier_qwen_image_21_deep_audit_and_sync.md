# Phase 6 Review 40 — Deep Architectural Audit & Ground-Truth Synchronization: Sketch Tier vs. Qwen-Image-2.1

## 1. Executive Summary & Training Architecture Contract

This canonical review establishes the ground-truth architectural contract across the Krisna monorepo, resolving all defects, aligning code, tests, and documentation with the official HuggingFace `diffusers` implementation of `Qwen/Qwen-Image-2.1`.

### A. Exactly Which Models Are We Training?
In Krisna, **only ONE model family is trained**:
1. **`Qwen/Qwen-Image-2.1` Polish Default LoRA / DoRA**:
   - Pinned config: [`training/configs/polish_qwen_image21_lora.yaml`](file:///d:/Krisna/training/configs/polish_qwen_image21_lora.yaml).
   - Targets Single-Stream DiT attention and SwiGLU feed-forward projections on image-conditioned pairs (`cond_image_column="condition_image"`, `image_column="image"`).
2. **`Qwen/Qwen-Image-2.1` Direct Preference Optimization (DPO)**:
   - Pinned config: [`training/configs/polish_qwen_image21_dpo.yaml`](file:///d:/Krisna/training/configs/polish_qwen_image21_dpo.yaml).
   - Flow-matching direct preference alignment on human UI design pairs using curriculum-weighted preference loss.
3. **`Qwen/Qwen-Image-2.1` Consistency Flow Matching (CFM) Distillation**:
   - Pinned config: [`training/configs/cfm_distill_qwen_image21.yaml`](file:///d:/Krisna/training/configs/cfm_distill_qwen_image21.yaml).
   - Distills continuous ODE trajectories to 4–6 steps for interactive previews.

**All other tiers are strictly frozen or deterministic**:
- **Planner Tier**: `Qwen3.5-9B` is **frozen** (4-bit GGUF via `llama.cpp` under strict GBNF grammar constraints compiled from `PlannerResponse`).
- **Sketch Tier**: `krisna.layout.v1` is a **deterministic Zero-VRAM CPU component** (Deterministic Box-Model Collision Partitioner + TrueType Wireframe Renderer). Not a learned model.
- **Polish Quality Tier**: `Qwen/Qwen-Image-2.1` base model is **frozen zero-shot** (40-step ODE integration).
- **Critic / Verifier**: `VerifierStack` is a deterministic evaluation harness evaluating layout IoU, OCR readability, aesthetic score, and lock preservation.

---

## 2. Comprehensive Defect Analysis & Ground-Truth Resolutions

### Defect 1: LoRA Target Projections for Single-Stream DiT (Fix J Resolution & Count Assertion)
- **Root Cause & Checkpoint Anatomy**: Previous target lists either targeted legacy dual-stream modules (`add_q_proj`, etc.) or guessed `mlp.net.*`. In the real `Qwen-Image-2.1` checkpoint, each single-stream transformer block contains:
  - Attention projections: `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0` (with `attn.norm_q`, `attn.norm_k`).
  - SwiGLU MLP: named `img_mlp`, containing linear projections `img_mlp.gate_layer`, `img_mlp.proj`, and `img_mlp.out`.
  Targeting `mlp.net.*` matched 0 MLP layers and silently trained attention-only ($4 \times 32 = 128$ modules) because PEFT silently ignores unmatched strings if at least one matches.
- **Resolution**:
  - Pinned exact PEFT targets across all configs (`polish_qwen_image21_lora.yaml`, `polish_qwen_image21_dpo.yaml`, `train_dpo.py`, `test_phase3_dpo.py`, and ablations):
    `to_q, to_k, to_v, to_out.0, img_mlp.gate_layer, img_mlp.proj, img_mlp.out`
  - Modules per block: 7 projections $\times 32$ transformer blocks $= 224$ total targeted layers.
  - Added strict post-adapter assertion in [`train_dpo.py`](file:///d:/Krisna/training/src/krisna_training/polish/train_dpo.py):
    Verifies that every target string in `lora_target_modules` matches and that total adapted layers exactly equals $7 \times 32 = 224$ modules, raising `RuntimeError` immediately on any mismatch.
  - Added meta-device diagnostic script: [`scripts/inspect_qwen_transformer_modules.py`](file:///d:/Krisna/scripts/inspect_qwen_transformer_modules.py) which verifies the 7 targets and 224 module count with zero VRAM and zero weight downloads.

### Defect 2: Refinement Mode Dual-Image Visual Conditioning & Training Gap
- **Root Cause**: The initial turn conditioned on the wireframe, but refinement previously conditioned only on the prior render with `"PRESERVE UNCHANGED"`. When refinement changed the layout (add, move, resize), that geometry reached the diffusion model only through prose. Moreover, prior-render-only editing was untrained.
- **Resolution**:
  - **Dual-Image Visual Conditioning**: In [`flows.py`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py) and [`polish_backend_qwen.py`](file:///d:/Krisna/inference/src/krisna_inference/backends/polish_backend_qwen.py), refinement mode conditions on **both images in fixed, deterministic block-causal order**:
    `image = [updated_wireframe_img, prior_render_img]`
    - Index 0 (`updated_wireframe_img`): injects the new geometry (bounding boxes, layout partitions, new component text).
    - Index 1 (`prior_render_img`): injects prior visual textures, colors, and locked photorealistic assets.
  - **Sequence Token Budget**:
    $2 \times 4096$ condition image tokens ($8,192$) $+ 4096$ target latents $+ \le 512$ text prompt tokens $\approx 12,800$ tokens ($\approx 12.8\text{K}$ tokens).
  - **Wireframe Re-Rendering**: When refinement actions modify the layout, `iterative_refine` immediately re-renders the updated wireframe and saves it to `state.layout.wireframe_ref` before requesting finalization.

### Defect 3: Monorepo Test Accounting & VRAM Realism
- **Test Suite Accounting**:
  - `tests/inference` + `tests/training`: **403 passed, 9 skipped**.
  - `tests/data_forge`: **264 passed, 2 skipped**.
  - Total across monorepo: **667 passed, 11 skipped, 0 failed**.
  - Zero tests disappeared. An earlier report ran only `tests/inference tests/training` (398 tests at the time) without `tests/data_forge` (264 tests). Both suites are fully tracked and passing.
- **VRAM Realism & Sequential Swapping**:
  - All VRAM profiles are strictly labeled as **"Documented Architecture Estimates / Unmeasured (Code-Level Invariants)"** pending live target hardware validation.
  - *12GB GPU Constraint*: Planner (5.5–6 GB 4-bit) and Polish (5.5–6 GB 4-bit NF4 with CPU offload) cannot reside concurrently on a 12GB GPU. Krisna's `SwapOrchestrator` enforces **sequential model swapping** (unloading Planner before loading Polish), requiring adequate host system RAM ($\ge 32\text{GB}$) and incurring PCI-e transfer latency.

### Defect 4: Grounded Regional Linguistic Feedback in Retry Prompt
- **Root Cause**: The previous retry prompt `"CRITICAL LAYOUT ENFORCEMENT: Strictly preserve all bounding box coordinates..."` referenced numbers the diffusion prompt did not contain, providing no actionable spatial signal.
- **Resolution**:
  - Implemented `build_grounded_spatial_retry_prompt()` in [`flows.py`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py).
  - Inspects `state.layout.layout_ref` nodes and translates geometric/OCR drift into clear, natural language regional instructions:
    `"LAYOUT CORRECTION: the 'Pay Now' button must sit at the bottom-right under the main card, the 'Checkout' header must sit at the top without downward drift, and all text labels must be rendered crisply with high contrast."`

### Defect 5: Seam Score Gating & Locked-Locked Collision Surfacing
- **Seam Score Threshold & Gating**:
  - Pinned explicit acceptance target `"seam_continuity": 0.85` in `EVALUATION_TARGETS` inside [`verifier_stack.py`](file:///d:/Krisna/inference/src/krisna_inference/verifiers/verifier_stack.py).
  - In [`flows.py`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py), if `seam_score < 0.85` on locked turns, it is added to `drift_reasons` and triggers the grounded layout retry engine.
- **Locked-Locked Node Collisions**:
  - Added `diagnostics: list[str] = Field(default_factory=list, max_length=64)` to `LayoutIR` in [`layout_ir.py`](file:///d:/Krisna/inference/src/krisna_inference/layout_ir.py).
  - [`layout_baseline.py`](file:///d:/Krisna/inference/src/krisna_inference/layout_baseline.py) detects overlapping locked nodes and records explicit conflict messages in `layout.diagnostics`.
  - [`flows.py`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py) surfaces these diagnostics directly into `state.conversation_history` as `[Layout Conflict]: Locked node 'A' collides with locked 'B'`, granting the Planner full visibility on the next turn.

### Defect 6: Planner $\leftrightarrow$ Polish Connection & Verbatim Label Guard
- **Root Cause**: `synthesize_polish_prompt` previously emitted a raw keyword list (`"components: ..., text labels: ..."`). Qwen-Image-2.1 expects long, descriptive prose generated by its Prompt Enhancement (PE) tooling.
- **Resolution**:
  - Implemented `synthesize_descriptive_polish_prompt()` in [`flows.py`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py):
    Generates rich, cohesive descriptive prose matching Qwen PE format, describing top header regions, central card hierarchy, bottom action bars, and palette tokens in natural language.
  - Implemented `assert_and_inject_verbatim_labels()` (Verbatim Label Guard):
    Asserts that 100% of required text labels from `LayoutIR` survive in the prompt. If any label was dropped during rewriting or generation, the guard re-injects it cleanly in an explicit typographic directive.
  - Documented A/B testing plan: compare the frozen Planner with PE system prompt against the external PE model.

### Defect 7: Font Trade-Offs & Empirical Floor Clarification
- **DejaVu Sans**: Bundled permissive font (`DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`) ensures deterministic, machine-independent rasterization and font stepping. While differing from modern UI sans-serifs (Inter, Roboto), it is a verified deterministic baseline; typography silhouette ablations are scheduled in the ablation roadmap.
- **12px Floor**: The 12px minimum size across typographic roles is an empirical legibility guideline ensuring clean character rendering on web/mobile viewports, not an absolute mathematical consequence of 16x VAE downsampling.

---

## 3. Comprehensive Verification Status Matrix

| Component | Architecture / Pinning | Status | Evidence / Test |
|---|---|---|---|
| **LoRA Targets** | `to_q,to_k,to_v,to_out.0,img_mlp.gate_layer,img_mlp.proj,img_mlp.out` | **Code-Level Invariant** | [`scripts/inspect_qwen_transformer_modules.py`](file:///d:/Krisna/scripts/inspect_qwen_transformer_modules.py) (224 modules); [`test_phase3_dpo.py`](file:///d:/Krisna/tests/training/test_phase3_dpo.py) |
| **LoRA Count Assertion** | Assert `len(matched) == 224` | **Code-Level Invariant** | [`train_dpo.py:L446-465`](file:///d:/Krisna/training/src/krisna_training/polish/train_dpo.py) fails fast if module count != 224 |
| **Refinement Conditioning** | Dual-Image `[wireframe, prior_render]` | **Code-Level Invariant** | [`polish_backend_qwen.py:L477-486`](file:///d:/Krisna/inference/src/krisna_inference/backends/polish_backend_qwen.py); [`flows.py:L599-610`](file:///d:/Krisna/inference/src/krisna_inference/orchestrator/flows.py) |
| **Verbatim Label Guard** | `assert_and_inject_verbatim_labels` | **Unit-Tested** | [`test_verbatim_label_guard_reinjects_missing_labels`](file:///d:/Krisna/tests/inference/test_tier_synchronization_and_rgba.py) |
| **PE Prose Prompt** | `synthesize_descriptive_polish_prompt` | **Unit-Tested** | [`test_synthesize_descriptive_polish_prompt_produces_pe_prose`](file:///d:/Krisna/tests/inference/test_tier_synchronization_and_rgba.py) |
| **Grounded Retry Feedback** | Region-specific linguistic instructions | **Unit-Tested** | [`test_grounded_spatial_retry_prompt_linguistic_feedback`](file:///d:/Krisna/tests/inference/test_tier_synchronization_and_rgba.py) |
| **Seam Gating** | `seam_continuity >= 0.85` | **Unit-Tested** | [`test_seam_continuity_evaluation_target_gating`](file:///d:/Krisna/tests/inference/test_tier_synchronization_and_rgba.py); `verifier_stack.py` |
| **Locked Conflict Surfacing** | Surface conflicts to Planner history | **Unit-Tested** | [`test_locked_locked_conflicts_surfaced_to_layout_diagnostics`](file:///d:/Krisna/tests/inference/test_tier_synchronization_and_rgba.py) |
| **Monorepo Test Suite** | 667 passed, 11 skipped, 0 failed | **Fully Verified** | PyTest across `tests/inference`, `tests/training`, `tests/data_forge` |
| **Zero C: Downloads** | `cache_dir` isolation verified | **Fully Verified** | `C:\Users\priya\.cache\huggingface\hub` does not exist (`False`) |

---

## 4. Next Steps & Execution Roadmap

1. **Live Model Turn & Hardware Profiling**: Execute first live run of `Qwen/Qwen-Image-2.1` on target GPU to measure sequential swap latencies and NF4 memory headroom.
2. **Drift-First Ablation Matrix**: Execute the 4-dimensional layout ablation benchmark prioritizing LayoutIR-to-output drift as the primary score.
3. **PE Model A/B Test**: Profile the frozen Planner with PE prompt formatting against the official Qwen Prompt-Enhancer model to evaluate memory-vs-fidelity trade-offs.
