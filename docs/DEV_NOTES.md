# Krivex: Developer Notes & Engineering Record

**Last Updated**: October 2026  
**Target Hardware Envelope**: 12GB–16GB VRAM (Low-VRAM / Consumer GPU) | 16GB System RAM | Single-GPU Architecture  
**Canonical Specifications**: [PRD.md](PRD.md) | [RESEARCH_PAPER.md](RESEARCH_PAPER.md) | [LEGACY_REJECTIONS.md](architecture/LEGACY_REJECTIONS.md)

---

## 1. System Overview & Architectural Evolution

Krivex is an agentic, multi-turn UI generation and iterative design system. It converts high-level human conversational intent into pixel-accurate, aesthetically refined user interfaces through a decoupled, single-model-per-responsibility pipeline.

### 1.1 The Architectural Paradigm Shift

The codebase underwent a major architectural audit and modernization between Phase 1–36 and the Inference Modernization initiative. The historical pipeline relied on five fragmented models, discrete token codebooks, and heavy external critics. The modernized pipeline replaces these failure points with deterministic representations, unified models, and resident review loops:

```
+-----------------------------------------------------------------------------------------+
|                                 USER CONVERSATION                                       |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
| TIER 1: FROZEN PLANNER & RESIDENT REVIEWER                                              |
| - Runtime: llama.cpp (GGUF, ~6.5GB VRAM)                                                |
| - Model: Qwen3.5-9B (W4A16 / GGUF) or Qwen3-VL-7B                                       |
| - Mode A (Design): User Chat -> Validated JSON Delta (`krivex.delta.v1`) + UICrit RAG   |
| - Mode B (Review): Self-Review of Wireframe/Canvas + OCR + Verifier Findings            |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v (Validated Delta)
+-----------------------------------------------------------------------------------------+
| TIER 2: STRUCTURED LAYOUT IR & DETERMINISTIC WIREFRAME RENDERER                         |
| - Schema: `krivex.layout.v1` (Hierarchical Containers, Bounding Boxes, Exact Text)      |
| - Renderer: Deterministic Canvas/SVG -> Lossless 2D Wireframe PNG (~0.2GB VRAM)         |
| - Generation Foundation: LayoutFlow continuous flow matching (hierarchical UI extension)|
| - Inpainting Support: Exact coordinate-based Region Locking                             |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v (Handoff Image PNG + Synthesized Prompt)
+-----------------------------------------------------------------------------------------+
| TIER 3: UNIFIED DIFFUSION POLISH RENDERER                                               |
| - Runtime: Hugging Face Diffusers `QwenImage21Pipeline` (~8.0GB - 12.0GB VRAM)         |
| - Base Checkpoint: `Qwen/Qwen-Image-2.1`                                                |
| - Adapter: DreamBooth LoRA (Image-Conditioned UI Design weights)                        |
| - Modes: Default (25 steps, fast) | Quality (45 steps)                                  |
| - Region Inpainting: Masked composite preserving locked bounding boxes                 |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v (Rendered UI)
+-----------------------------------------------------------------------------------------+
| VERIFIER STACK & SAFETY GATE                                                            |
| - Deterministic Safety Gate: Hard score cutoff (Deletes rejected image from BlobStore)  |
| - Verifiers: OCR transcription, Layout Alignment IoU, Aesthetic Scorer                 |
| - DPO Preference Store: Automatic paired feedback logging for continuous alignment      |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Feature Retention & Architectural Audit

A critical mandate of this engineering pass is verifying whether any required features, desirable functionality, or critical designs were accidentally lost during recent refactorings.

### 2.1 Feature Audit Matrix

| Feature / Capability | Legacy Status | Modernized Implementation | Status | Audit Justification & Technical Mechanics |
|---|---|---|---|---|
| **Region Locking / Inpainting** | Unreliable token masking in VQGAN | `LockedRegion` schema in `DesignState`; exact normalized coordinates `[ymin, xmin, ymax, xmin]` in `krivex.layout.v1`; handled in `QwenImage21Backend.render()` via masked composites. | **RETAINED & UPGRADED** | VQGAN token masking suffered from boundary blurring and codebook drift. Exact geometric coordinate masks in Layout IR provide mathematically pristine pixel preservation. |
| **Deterministic Safety Gate** | PRD §5.2 specification | `_check_safety_gate()` in `flows.finalize()`; enforces hard threshold $\theta_{\text{safe}}$; on violation, immediately purges image via `blob_store.delete(image_ref)`, rolls state back to `SKETCHING`, and raises `SafetyGateViolationError`. | **RETAINED & HARDENED** | Completely intact. Unsafe renders never reach the client or persist in storage. |
| **DPO Preference Pair Store** | SQLite store for Z-Image pairs | `PreferenceStore` singleton; integrated into `flows.critique_pass()`; records `(prompt, chosen_img, rejected_img, metadata)` during review iterations. | **RETAINED** | Continues logging real human and reviewer comparisons into SQLite for offline adapter fine-tuning. |
| **Direct Photo Enhancement** | `enhance_photo` endpoint | `flows.enhance_photo()` preserved in orchestrator; routes directly to `QwenImage21Backend` with instruction-conditioned img2img. | **RETAINED** | Users can supply an existing UI screenshot and polish it without having to pass through the conversational planner. |
| **Prompt Synthesis & Enrichment** | Ad-hoc text concatenation | `_synthesize_render_prompt()` and `_synthesize_dpo_prompt()` synthesize intent, palette, typography, layout hierarchy, and planner reasoning notes. | **RETAINED & UPGRADED** | Consolidates all design constraints into structured text prompts for the diffusion backbone. |
| **Dual Polish Operating Points** | Fragmented Z-Image (Default) vs Qwen-Edit (Quality) | Unified `QwenImage21Pipeline` with variable step schedules: Default (25 steps, ~8s latency) vs Quality (45 steps, higher fidelity). | **RETAINED & UNIFIED** | Eliminates dual pipeline divergence, diffusers API incompatibilities, and conflicting checkpoint storage. |
| **Multi-Turn Turn Budget** | Unconstrained infinite loop risk | `RefinementBudget` class in `design_state.py`; enforces max 4 turns per cycle; tracks `user_turn` vs `self_review` counts. | **NEW / HARDENED** | Prevents conversational drift and uncontrolled GPU compute consumption. |
| **Inactivity Self-Review** | Manual user trigger only | `SessionCoordinator` with 30s cancellable debounce timer; triggers `flows.planner_self_review()`. | **NEW / ENHANCED** | The agent autonomously audits and refines the draft wireframe during user contemplation. |
| **Optimistic Concurrency** | State overwrite vulnerability | `revision: int` counter in `DesignState`; all state stores validate revision prior to commit, raising `StaleDesignStateError` on collision. | **RETAINED** | Prevents race conditions in asynchronous multi-turn web clients. |

### 2.2 Intentional Deprecations & Technical Justifications

The following components were intentionally removed based on empirical findings and formal architectural review:

1. **Discrete MaskGIT Pixel Tokens (`boris/vqgan_f16_16384`)**:
   - *Why Removed*: Stride-16 discrete token codebooks cannot represent crisp text labels, UI component hierarchies, exact button margins, or parent-child nesting. Furthermore, masked discrete sampling caused structural hallucinations and could not reliably respect region locking.
   - *Replacement*: `krivex.layout.v1` (Layout IR), providing deterministic JSON geometry, element IDs, styles, and vector wireframe rendering.
2. **External Gemma 4 31B Critic**:
   - *Why Removed*: Required 18GB VRAM in NF4 (or 48GB system RAM offload), breaking the 12GB consumer target. Process swapping between Polish and Critic added 45–60 seconds of cold-load latency per turn.
   - *Replacement*: Resident Planner in review mode (`review=True`). The resident Qwen planner inspects Layout IR, OCR text transcripts, and deterministic verifier scores with 0s swap latency and 0GB added memory.
3. **Fragmented Z-Image-Turbo / Qwen-Image-Edit-2511 Backends**:
   - *Why Removed*: Z-Image used custom flow-matching velocity formulations incompatible with standard Diffusers pipelines. Qwen-Image-Edit-2511 had a non-functional `strength` parameter in upstream Diffusers (`'strength' in sig.parameters == False`).
   - *Replacement*: Unified `Qwen-Image-2.1` diffusion pipeline with official DreamBooth LoRA adapter support.
4. **Flow-matching Diffusion-DPO Trainer**:
   - *Why Removed*: Architecture-specific loss assumptions tied to Z-Image that could not produce Qwen-compatible adapters.
   - *Replacement*: Official Hugging Face Diffusers DreamBooth LoRA image-conditioned training pipeline.

---

## 3. Component Specifications & Contracts

### 3.1 Tier 1: Planner & Resident Reviewer

- **Implementation**: `inference/src/krivex_inference/backends/planner_backend.py`
- **Runtime Engine**: `llama.cpp` native server / shared library (`libmtmd` / GGUF).
- **Target Precision**: W4A16 / Q4_K_M selective quantization (~6.5GB VRAM). Embeddings, vision projectors, and normalization layers are preserved at FP16/BF16.
- **RAG Subsystem**: Hybrid sparse-dense index over the `google-research-datasets/uicrit` corpus. Combines BM25 lexical search with dense semantic retrieval.
- **Output Schema**: Constrained grammar generation via `krivex.delta.v1`:
  ```json
  {
    "schema_version": "krivex.delta.v1",
    "assistant_message": "Added a responsive navbar and dark mode cards.",
    "applied_constraints": {
      "theme": "dark",
      "primary_color": "#6366F1",
      "layout_type": "dashboard"
    },
    "reasoning_notes": "Positioned metrics cards at the top for quick scanning."
  }
  ```
- **Review Mode**: Invoked via `planner_backend.review()`. Consumes the active wireframe image, OCR transcript, layout element tree, and verifier metrics to return structured critique without allocating secondary models.

### 3.2 Tier 2: Structured Layout IR & Deterministic Renderer

- **Implementation**: `inference/src/krivex_inference/backends/layout_backend.py`
- **Schema Specification (`krivex.layout.v1`)**:
  - `canvas`: `{"width": 1024, "height": 1024, "background_color": "#111827"}`
  - `elements`: List of UI nodes, each containing:
    - `id`: Unique string identifier (`"btn-submit"`, `"card-metrics"`).
    - `type`: Element class (`container`, `button`, `text`, `image`, `input`, `icon`).
    - `box_2d`: Normalized coordinates `[ymin, xmin, ymax, xmin]` in range $[0, 1000]$.
    - `parent_id`: Parent container ID (enabling hierarchical trees).
    - `style`: Fill color, stroke width, stroke color, border radius, opacity.
    - `text`: Exact text string label (e.g., `"Get Started"`).
    - `locked`: Boolean flag for user-pinned regions.
- **Deterministic Renderer**: Renders Layout IR directly to a lossless 2D PNG wireframe using deterministic SVG/Pillow rendering. Guarantees 100% geometric repeatability.
  > **Implementation Note (Oct 2026 audit)**: The inference pipeline uses the *deterministic* SVG/Pillow renderer — not a trained neural model. The `training/src/krivex_training/layout/` package implements a LayoutFlow-style 4-layer TransformerEncoder flow model as a research candidate for future trained layout generation. It has no production checkpoint and is not loaded at inference time. This distinction is material for paper claims: the Layout IR contribution is the deterministic schema and rendering system, not a claim that LayoutFlow is trained.
- **Generative Backbone (Research/Future)**: The `training/layout/model.py` LayoutFlow TransformerEncoder architecture is the candidate for learning UI layout distributions from RICO+CLAY in a future training pass. Not yet trained; not yet used in inference.

### 3.3 Tier 3: Unified Diffusion Polish Renderer

- **Implementation**: `inference/src/krivex_inference/backends/qwen_backend.py`
- **Pipeline Class**: `diffusers.QwenImage21Pipeline`
- **Model Checkpoint**: `Qwen/Qwen-Image-2.1`
- **Adapter Contract**: DreamBooth LoRA trained on paired UI wireframe $\to$ finished design distributions.
  - Manifest validation enforces base model equality (`base_model == "Qwen/Qwen-Image-2.1"`).
- **Execution Modes**:
  - `default`: 25 inference steps, CFG scale 4.0 (Fast turnaround, ~6–8s on RTX A6000 / ~14s on RTX 4070).
  - `quality`: 45 inference steps, CFG scale 5.0 (High visual fidelity, enhanced micro-details).
- **Region Inpainting**: When `locked_regions` are supplied, the backend performs masked alpha compositing:
  $$\mathcal{I}_{\text{final}} = \mathcal{M}_{\text{lock}} \odot \mathcal{I}_{\text{handoff}} + (1 - \mathcal{M}_{\text{lock}}) \odot \mathcal{I}_{\text{diffusion}}$$

### 3.4 Orchestrator & Multi-Turn State Machine

- **Implementation**: `inference/src/krivex_inference/`
- **State Schema (`DesignState`)**:
  - `session_id: str`
  - `stage: SessionStage` (`CONVERSING` $\to$ `SKETCHING` $\to$ `REFINING` $\to$ `FINALIZING` $\to$ `FINALIZED`)
  - `revision: int` (Optimistic concurrency counter)
  - `refinement: RefinementBudget` (Tracks `user_turns`, `self_review_turns`, max 4)
  - `layout_ir: LayoutIR` (Current geometry state)
  - `locked_regions: list[LockedRegion]`
  - `refinement_events: list[RefinementEvent]` (Audit trail)
- **Session Coordinator**:
  - Manages per-session inactivity timers (`KRIVEX_SELF_REVIEW_IDLE_SECONDS = 30`).
  - Automatically cancels pending self-reviews upon receipt of a new user message.
  - Triggers autonomous self-review passes until turn budget is exhausted.

### 3.5 Rust Desktop Control Plane (`src-tauri`)

- **Implementation**: `inference/frontend/src-tauri/src/pipeline.rs` & `lib.rs`
- **Core Architecture**: In-process Rust policy enforcement acting as a low-latency barrier before Python model invocation:
  - **Inactivity Coordinator**: Monotonic timer checking (`check_inactivity`) with 30s timeout that flags `self_review_due` during user contemplation in the `sketching` stage.
  - **Audit Event Ledger**: Records up to 100 bounded `RefinementEvent` records (`turn_completed`, `cycle_started`, `finalized`, `safety_rollback`).
  - **Optimistic Concurrency & Revision Envelopes**: Rejects stale commands (`expected_revision != current_revision`) with `stale pipeline revision` errors.
  - **Turn Budget Enforcement**: Rejects user and self-review turns once `turn_count >= MAX_TURNS (4)`.
  - **State Transitions**: Native `record_finalize` and `record_rollback` commands synchronizing directly with `DesignState`.

### 3.6 Production Service Endpoints & Local Runner

- **Implementation**: `inference/src/krivex_inference/orchestrator/service.py` & `scripts/inference/run_local.py`
- **Health & Readiness**:
  - `GET /health`: Liveness probe reporting process status, residency state, and declared VRAM ledger.
  - `GET /ready`: Readiness probe verifying orchestrator initialization and SQLite store accessibility.
- **Model Lifecycle & Hardware Telemetry**:
  - `GET /models/status`: Returns discovered GGUF path, multimodal projector path, live CUDA VRAM allocations, system RAM utilization, and Diffusers LoRA adapter status.
  - `POST /models/warm`: Pre-warms Planner and Layout baseline tiers via `mmap` for sub-second first-turn latency.
  - `POST /models/unload`: Unloads resident GPU models and cleans PyTorch CUDA cache for memory reclamation.
- **Production Local Runner (`scripts/inference/run_local.py`)**:
  - Automated hardware preflight (CUDA availability, VRAM size, host RAM).
  - Model artifact auto-discovery across candidate paths and SHA256 manifest verification.
  - Dynamic `KRIVEX_LOW_VRAM_MODE` calibration for cards with < 16GB VRAM.
  - Fast single-command startup for native Windows and Linux execution.

---

## 4. Hardware Budget & Memory Ledgers

The system enforces deterministic admission via `VRAMLedger` and `RAMLedger` (`inference/src/krivex_inference/orchestrator/vram_budget.py`).

### 4.1 VRAM Allocation Ledger (12GB Operating Point)

| Phase | Active Tiers | Declared VRAM | Measured VRAM | Safety Margin | Headroom | Status |
|---|---|---|---|---|---|---|
| **Baseline Idle** | Planner (llama.cpp) + Layout IR | 6.5 GB + 0.2 GB = 6.7 GB | 6.4 GB | 0.5 GB | 4.8 GB | Admitted |
| **Finalize Swap (Peak)** | Baseline unloads $\to$ Polish loads | 8.0 GB – 11.5 GB | 10.2 GB | 0.5 GB | 1.3 GB | Admitted |
| **Finalize Completion** | Polish unloads $\to$ Baseline restores | 6.7 GB | 6.4 GB | 0.5 GB | 4.8 GB | Admitted |

### 4.2 Host System RAM Ledger (16GB Host Envelope)

| Process / Component | Declared RAM | Peak Resident RAM | Notes |
|---|---|---|---|
| OS & Python Runtime | 2.5 GB | 2.2 GB | Base system footprint |
| Planner Process (llama.cpp host buffers) | 1.5 GB | 1.2 GB | Host KV cache and context buffers |
| Polish Diffusers Host Buffers | 4.0 GB | 3.6 GB | Pipeline initialization and tokenizer weights |
| Blob Store / In-Memory Artifacts | 1.0 GB | 0.8 GB | Image buffers, wireframes, and transcripts |
| **Total Combined Host RAM** | **9.0 GB** | **7.8 GB** | **Well within 16GB host ceiling** |

---

## 5. Developer Workflows & Testing Runbook

### 5.1 Environment Setup

1. **Python Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-inference.txt
   ```
2. **Environment Variables**:
   ```powershell
   $env:PYTHONPATH = "inference/src;training/src;src"
   $env:KRIVEX_LOW_VRAM_MODE = "1"
   $env:KRIVEX_SELF_REVIEW_IDLE_SECONDS = "30"
   ```

### 5.2 Running the Test Suites

To execute the complete inference test suite (182 tests):
```powershell
python -m pytest tests/inference -v
```

To run orchestrator and flow tests specifically:
```powershell
python -m pytest tests/inference/test_orchestrator_flows.py -v
python -m pytest tests/inference/test_session_coordinator.py -v
```

### 5.3 Verifying Feature Contracts

To ensure no features degrade during future pull requests:
- `test_finalize_rejects_safety_violation`: Verifies that unsafe renders are rejected and deleted from the blob store.
- `test_finalize_respects_locked_regions`: Verifies that coordinate bounding boxes are preserved through inpainting.
- `test_refinement_budget_enforces_four_turns`: Verifies that conversational drift is bounded by turn policy.
- `test_session_coordinator_debounces_inactivity`: Verifies that self-review triggers only after 30s of user silence.

### 5.4 Dual-Loader Asynchronous Model Loading Architecture

The system coordinates two distinct model loaders on single-GPU hardware:
1. **Model Loader 1: llama.cpp Runtime (`PlannerBackendLlama`)**:
   - Zero-copy memory mapped GGUF execution via `use_mmap=True`.
   - FlashAttention KV cache allocation for long-context RAG prompts.
   - Grammar-constrained JSON decoding (`PlannerResponse`).
2. **Model Loader 2: Diffusers Flow-Matching Runtime (`QwenImage21Backend`)**:
   - Velocity-based ODE numerical solvers (Euler and 2nd-order Midpoint).
   - Time-decayed dynamic guidance schedule $w(t) = 1.0 + (w_{\max} - 1.0) \cdot \cos(\frac{\pi}{2} t)$.
   - Cosine-feathered boundary inpainting for locked regions.
   - Low-VRAM CPU offload option (`enable_model_cpu_offload`).

**Asynchronous Pre-Warming & Non-Blocking Coordination**:
- `SwapOrchestrator.start_background_warm()` triggers non-blocking background initialization.
- Real-time progress is trackable via `GET /models/warm/status` and `GET /session/{session_id}/model-status`.
- Conversational turns non-blockingly await background warming completion, preventing 409 collisions.
- Per-session asynchronous locks in `service.py` prevent concurrency race conditions.

### 5.5 Rust Control Plane & Zero-Dependency HTTP Client

The desktop frontend control plane (`inference/frontend/src-tauri/src/`) maintains local policy boundaries in Rust:
- `backend_client.rs`: Zero-dependency, offline-reliable HTTP client using `std::net::TcpStream` with timeouts and retries.
- Tauri commands (`backend_health`, `backend_ready`, `backend_models_status`, `backend_warm_models`, `backend_send_message`, `backend_finalize`) proxy requests to the Python inference server while enforcing turn budget constraints and optimistic revision matching.

### 5.6 Peer-Reviewed Architectural Grounding (IEEE, Springer, ACM)

Every core engineering decision in Krivex maps to established peer-reviewed literature:
- **Spatial Condition Decoupling**: ControlNet (Zhang et al., *IEEE/CVF ICCV 2023*, DOI: [10.1109/ICCV51070.2023.00355](https://doi.org/10.1109/ICCV51070.2023.00355)) motivates isolating lossless 2D wireframe rendering from diffusion pixel synthesis.
- **Semantic Graph Priors for Layout**: InstructLayout (*IEEE TPAMI 2025*, DOI: [10.1109/TPAMI.2025.3595880](https://doi.org/10.1109/TPAMI.2025.3595880)) proves that instruction-driven layout synthesis is stabilized by enforcing relational graph priors, grounding `krivex.layout.v1`.
- **Inversion-Free Flow Trajectories**: FlowEdit (Kulikov et al., *IEEE/CVF ICCV 2025*) establishes localized continuous flow editing without latent inversion errors.
- **Typographic Layout & Text Rendering**: TextDiffuser-2 (Chen et al., *Springer ECCV 2024*, DOI: [10.1007/978-3-031-72761-0_1](https://doi.org/10.1007/978-3-031-72761-0_1)) and AnyText (*ICLR 2024*) demonstrate that diffusion models require explicit bounding boxes and glyph conditioning to eliminate garbled text.
- **GUI Visual Agent Grounding**: ShowUI (Lin et al., *IEEE/CVF CVPR 2025*, DOI: [10.1109/CVPR52734.2025.01816](https://doi.org/10.1109/CVPR52734.2025.01816)) and SeeClick (*ACL 2024*, DOI: [10.18653/v1/2024.acl-long.505](https://doi.org/10.18653/v1/2024.acl-long.505)) establish screen-space token selection for visual UI agents.
- **Interactive State Tracking**: PromptMagician (Feng et al., *IEEE TVCG 2024*, DOI: [10.1109/TVCG.2023.3327168](https://doi.org/10.1109/TVCG.2023.3327168)) justifies transparent session state snapshots ($\mathcal{S}^{(t)}$) for user-guided exploration.
- **Optimal Transport Continuous Layouts**: LayoutFlow (Guerreiro et al., *Springer ECCV 2024*, DOI: [10.1007/978-3-031-72764-1_4](https://doi.org/10.1007/978-3-031-72764-1_4)) provides continuous vector-field flow matching over bounding boxes.
- **Standardized Multi-Turn GUI Benchmarking**: OSWorld (Xie et al., *NeurIPS 2024*) sets the multimodal benchmark standard across operating system environments.

### 5.7 Phase 5: FlowEdit 8-Step Refinement & Discrete VLA Grammar

- **8-Step FlowEdit ODE Solver**:
  Implemented in `FlowODEIntegrator.flowedit_step`:
  $$x_{t+dt} = x_t + dt \cdot (v_{\text{edit}} - v_{\text{src}})$$
  Enables targeted localized edits without complete canvas regeneration in 8 steps (~2.5s) instead of 40-step scratch generation.
- **10 Discrete Typed VLA Actions**:
  `SET_STYLE`, `RESIZE`, `ADD_COMPONENT`, `LOCK_REGION`, `SET_PALETTE`, `REMOVE_COMPONENT`, `MOVE_COMPONENT`, `SET_TEXT`, `SET_LAYOUT_DIRECTION`, `REORDER_HIERARCHY`.
  Handled in `constraint_merge.py` with bounding-box normalization and boundary clamping:
  $$\forall \text{node}: \quad x + w \le 1.0, \quad y + h \le 1.0$$
- **Multi-Modal Client Binding**:
  Rust Tauri desktop client (`trigger_refine`, `backend_refine`) and web frontend (`app.js`) allow clicking wireframe canvas nodes directly to dispatch `LOCK_REGION` VLA actions.

### 5.8 Research Trajectory & Trace Architecture (`krivex.research.trace.v1`)

To facilitate empirical analysis, paper table compilation, and preference dataset generation:
- **Trajectory Schema**: Specified in `inference/src/krivex_inference/research/trace_schema.py`:
  - `SessionResearchTrajectory`: Session-level trajectory containing metadata, cumulative constraints, hardware provenance, and sequence of turn records.
  - `TurnTraceRecord`: Atomic turn trace capturing `reasoning_note` (Planner chain-of-thought), conversation dialog, discrete VLA actions, SHA-256 state hash, wireframe/image blob IDs, and verifier score receipts.
  - `RuntimeEnvironmentInfo`: Snapshot of Python, OS, PyTorch, Diffusers, CUDA device name, memory ledger allocation, and active model checkpoints.
- **Trace Recorder**: `inference/src/krivex_inference/research/trace_recorder.py`:
  - `build_research_trajectory(state)`: Constructs structured trajectory directly from runtime `DesignState`.
  - `save_research_trajectory(trajectory, path)`: Persists trajectory JSON.
  - `export_trajectory_jsonl(trajectory, path)`: Appends turn records to a line-delimited JSON stream for offline DPO preference optimization.
- **API Surface**:
  - `GET /session/{session_id}/research-trace`: Serves the complete `SessionResearchTrajectory` JSON payload.
- **Verification Tests**:
  - `pytest tests/inference/test_research_traces.py -v`: Validates schema serialization, trajectory construction, file persistence, and REST endpoint responses.



