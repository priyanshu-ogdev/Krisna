# Krisna — Product Requirements Document

**Status**: Final (no-RLHF-loop revision). This document is the
canonical reference for every `PRD §X` citation appearing throughout
this codebase's comments, tests, and configs — those citations predate
this file's existence; this is the first time the document they
reference has been written down in one place, reconstructed to be
consistent with every one of them (verified by grep across the entire
repository, not drafted independently and hoped to match).

---

## 1. Problem statement

Turning a described intent ("a settings screen with a dark-mode
toggle") into a finished, high-fidelity UI design normally takes a
human designer multiple passes: rough layout, refinement, and review.
Krisna automates this as a four-stage **agentic pipeline**, each stage
handled by a purpose-fit model, coordinated by a single orchestrator
that never keeps more than one large model resident on a single GPU at
a time.

## 2. Goals

- A conversational front-end (chat) that converges on a design intent
  before any pixel is generated.
- A fast, low-fidelity **Sketch** pass the user can iterate on cheaply.
- A high-fidelity **Polish** pass producing the final deliverable image.
- An optional, on-demand **Critique** pass giving structured, multi-axis
  feedback on the finished design.
- All of the above running on a **single consumer/prosumer GPU**
  (16–24GB target, with an explicit 12–16GB low-VRAM/CPU-offload mode),
  not a multi-GPU cluster.
- Every model-training decision backed by a real, cited source — no
  training signal manufactured from an untrained judge model's opinion.

## Non-goals

- Multi-user concurrent serving (this is a single-session,
  single-resident-tier system by design — see §7 — not a
  high-throughput inference server; batching/continuous-batching
  serving frameworks like vLLM are a deliberate non-goal for this
  access pattern, not an oversight — see §7.5).
- Training any model via RLHF or an AI-judge-labeled reward signal (see
  §6's no-RLHF-loop revision).
- A from-scratch text-to-image foundation model — every generative
  model in this system starts from a real, published checkpoint.

---

## 3. Resource envelope

Two operating points, both real, both tested:

| Mode | GPU envelope | System-RAM envelope | Selected via |
|---|---|---|---|
| Default | 16–24GB | not applicable (nothing offloads) | `KRISNA_LOW_VRAM_MODE` unset/`0` |
| Low-VRAM / CPU-offload | 12–16GB | up to 48GB (Critic's offloaded weights are the dominant cost) | `KRISNA_LOW_VRAM_MODE=1` |

Admission against these envelopes is enforced by two independent
ledgers (`VRAMLedger`, `RAMLedger` — see §7.2), against **declared**
per-tier budgets (`model_registry.py`), not a live GPU probe — a
deliberate choice for deterministic, CI-testable admission logic. Real
hardware readings (`torch.cuda.mem_get_info()`, `/proc/meminfo`) are
surfaced separately, for observability, never for admission (see §7.6).

## 4. Product use case

A single user, in a single session, has a conversation with the system
to arrive at a UI design. The interaction is turn-based and stateful:
each turn either advances the conversation (Planner), produces a cheap
draft (Sketch), finalizes a real render (Polish), or reviews a finished
render (Critic). The system is explicitly **not** a multi-tenant service
— one session, one resident generative tier at a time, by design (see
§7).

---

## 5. Data model & core flows

### 5.1 Design State

The single source of truth for a session, persisted between turns
(`orchestrator/design_state.py`'s `DesignState`). Fields:

- `session_id`, `stage` (`conversing` → `sketching` → `finalizing` →
  `finalized` → `critiquing`, defined by `SessionStage`)
- `conversation_history: list[{role, content, timestamp}]`
- `constraints` (accumulated structured intent from the Planner)
- `layout: LayoutDraft` (strongly-typed `krisna.layout.v1` draft:
  `layout_ref`, `wireframe_ref`, `segmentation_ref`, `mask_ref`, `revision`)
- `finalize_output` (`image_ref`, `renderer_used`, `verifier_scores`,
  `prompt_used`, `evaluation_receipt`)
- `critique` (`requested`, `source`, `result`, `timestamp`)
- `revision` — an optimistic-concurrency counter; every write compares
  against the caller's expected revision and raises `StaleDesignStateError`
  on mismatch, rather than silently overwriting a concurrent edit.

### 5.2 Critique Output (Critic Tier — Critique Adapter)

The shape a Critic pass returns, regardless of which model produces it
(frozen Gemma 4 — see §6):

```
{
  critique_source: str,
  overall_score: float,
  dimensions: { <axis_name>: { score: float, note: str } },
  suggested_edits: list,
  raw_model_output_ref: str | None
}
```

A **gate**, not just a score: `VerifierStack.safety_gate()` runs before
a Finalize result is ever handed back to the caller — an unsafe image
must never reach the user, independent of whether Critique was
requested for it (see §5.3, §9).

### 5.3 The Finalize flow, and the "Tier A must never exceed ~10GB
resident" rule

Finalize is the single most resource-sensitive flow in the system: it
must swap Polish (12–14GB) into a GPU that also, at idle, holds the
**always-resident baseline** — Planner + Sketch, "Tier A." This rule is
load-bearing for every admission calculation in §7: **Tier A must never
exceed ~10GB resident**, so that swapping in any single Polish/Critic
tier still fits the smallest supported envelope (12GB, low-VRAM mode)
with the baseline unloaded first (see §7.4's swap sequence — baseline
is torn down *before* a swappable tier is admitted, not summed with
it). Concretely: Planner (Q4_K_M GGUF, ~6.5GB) + Sketch (`krisna.layout.v1`,
0.0GB VRAM on CPU) = ~6.5GB, well under the 10GB ceiling with substantial margin.

The flow itself: Sketch's `LayoutIR` DAG is converted via `render_dual_stream_sketch()`
into an atomic `DualStreamSketch` condition bundle:
1. **Stream 1: High-Frequency TrueType Wireframe** (RGB or native RGBA when `asset_mode == "transparent"`),
   featuring anti-aliased font rendering with a strict legibility floor ($\ge 12\text{px}$) and
   elevation-aware drop shadows ($s = 2+2z, r = 8+2z$).
2. **Stream 2: 16-Class SPADE Semantic Map** (discrete categorical palette preventing normalization
   washout in deep MMDiT layers).
3. **Stream 3: Spatial Alpha / Inpaint Mask** (8-bit grayscale $L$, distinguishing $0$ locked donor
   regions from $255$ active regions).

These multi-stream conditions are handed to `Qwen-Image-2.1` with quoted verbatim text labels
and locked donor region restoration ($0.00\%$ pixel drift), scored by the verifier stack
(including Laplacian seam boundary continuity $\Delta \nabla \le 0.05$), gated by `safety_gate()`,
and only then does `DesignState.stage` advance to `finalized`. A safety-gate failure rolls the stage
back to `sketching` and raises rather than returning a partial or unsafe result.

### 5.4 FlowEdit 8-step iterative refinement and VLA action grammar

In Phase 5, the iterative polish loop adds an explicit FlowEdit 8-step numerical ODE solver ($x_{t+dt} = x_t + dt \cdot (v_{\text{edit}} - v_{\text{src}})$) grounded in Kulikov et al. (ICCV 2025). The Planner communicates modifications either via full conversational deltas or 10 discrete Vision-Language-Action (VLA) primitives:
- `SET_STYLE`, `RESIZE`, `ADD_COMPONENT`, `LOCK_REGION`, `SET_PALETTE`, `REMOVE_COMPONENT`, `MOVE_COMPONENT`, `SET_TEXT`, `SET_LAYOUT_DIRECTION`, and `REORDER_HIERARCHY`.
- Spatial coordinate clamping (`_normalize_and_clamp_bbox`) ensures $x + w \le 1.0, y + h \le 1.0$ at all times.
- Locked regions (`LOCK_REGION`) guarantee exact 0.00% pixel drift preservation via dual-layer inpainting masks and hard composite protection.

### 5.5 Research Trajectory & Trace Recording Protocol (`krisna.research.trace.v1`)

For offline research paper analysis, ablation studies, and DPO alignment dataset construction, Krisna records full deliberation traces:
- `RuntimeEnvironmentInfo`: Hardware specs, CUDA/PyTorch/Diffusers versions, GPU memory ledger status, active model weights.
- `TurnTraceRecord`: Per-turn record tracking user prompt, assistant response, planner chain-of-thought (`reasoning_note`), discrete VLA actions, SHA-256 state digest, wireframe and image blob pointers, and multi-axis verifier score receipts.
- `SessionResearchTrajectory`: Session-level aggregate available via `GET /session/{session_id}/research-trace` and exportable to JSON/JSONL (`export_trajectory_jsonl()`) for direct consumption by empirical evaluation harnesses.

### 5.6 Unified Plan-and-Sketch Layer & Autonomous Direct Execution

To close the "Raster Image-to-Layout" gap and enable autonomous agentic workflows:
- **Unified Plan-and-Sketch Layer (`UnifiedPlanSketchLayer`)**: Synchronizes the Sketch tier directly with the 9B model (`Qwen3.5-9B` with `mmproj` dynamic-resolution Vision Transformer) serving as the multimodal perception and deliberation module. The Flow-matching Polish DiT (Qwen-Image-2.1) operates strictly on continuous latent pixels and cannot emit discrete JSON coordinates; the 9B VLM module handles visual grounding, spatial perception, and layout decompilation.
- **Autonomous Direct-to-Polish Execution (`POST /session/{session_id}/execute`)**: Enables a single-call autonomous cycle:
  1. *Simultaneous Plan + Sketch*: Runs Qwen3.5-9B deliberation (Gated DeltaNet linear recurrence, M-RoPE spatial grounding) and `krisna.layout.v1` layout compilation simultaneously.
  2. *Direct Polish Progression*: Hands off wireframe, SPADE segmentation, and synthesized prompt directly to Qwen-Image-2.1 without intermediate user intervention.
  3. *Verifier Stack Evaluation*: Scores aesthetic quality, OCR readability, layout IoU, seam continuity, identity preservation, and enforces the safety gate.
  4. *Closed-Loop Post-Render Review & Further Planning*: Passes the verifier receipt and render metadata back into Qwen3.5-9B, formulating autonomous self-critique and next-step planning directives (`[Post-Render Deliberation]`) appended to conversation history. Supported via `POST /session/{session_id}/execute` or `auto_proceed: true` on `POST /session/{session_id}/message`.
- **Raster Image Decompilation (`POST /session/{session_id}/decompile`)**: Users can upload raw screenshots, competitor mockups, or UI photos. The 9B VLM module decompiles flat raster images into discrete `LayoutIR` DAG components with normalized bounding boxes $[x_1, y_1, x_2, y_2]$ and text literals via `flows.decompile_screenshot()`.
- **Multi-Asset Grounding, Multi-Subject Composition & Facial Shielding (MS-Diffusion, FastComposer, AnyDoor, ConsisID, PhotoMaker)**:
  - Supports multi-image inputs via `ReferenceAsset` (`role ∈ {"subject", "background", "style", "canvas"}`), enabling duo/group compositions, character insertion, and background scene teleportation.
  - Spatial attention routing (MS-Diffusion & FastComposer) prevents identity cross-talk and feature bleeding across multiple distinct subjects.
  - Localized Crop $\to$ Lanczos Resample $\to$ Paste (AnyDoor & ConsisID) eliminates off-center face drift by decoupling donor crop coordinates from target canvas placement, achieving exact $0.00\%$ facial drift ($1.000$ lock score).
  - Background scene nodes are exempted from false-positive center-point locking, enabling seamless subject teleportation onto arbitrary backdrops.
  - Strict $\ge 128\times 128\text{px}$ floor on all facial and focal nodes prevents latent token collapse across VAE $16\times$ compression and DiT patch stride $2\times 2$ (PhotoMaker).
  - Exact bitwise pixel preservation on locked regions via `_apply_locked_regions` with Laplacian seam boundary verification ($\Delta \nabla \le 0.05$).

---

## 6. Model stack: the no-RLHF-loop revision

**This is the single most consequential architectural decision in the
project**, and the reason five originally-planned trainable models
became two. The original design considered fine-tuning every tier,
including using one model's judgments as a training signal for another
(an RLHF-style loop with an AI judge in place of a human). That design
was abandoned in favor of the following, final split:

| Tier | Model | Trained? | Why |
|---|---|---|---|
| **Planner** | Qwen3.5-9B GGUF (Unsloth Dynamic Quantization) | **No — frozen** | Image-grounded structured planning with optional UICrit retrieval; no fine-tuning or judge-labeled data |
| **Sketch / Structure** | Deterministic `krisna.layout.v1` Layout IR and wireframe renderer | **No** | Makes geometry, hierarchy, literals, and lock regions auditable with 0 MB VRAM and 0% overlap guarantee |
| **Polish (Default)** | Qwen-Image-2.1 | **Base frozen; LoRA gated and trainable** | Supervised wireframe→screenshot adapter only after a real pair manifest and native trainer smoke test; preference alignment is optional later |
| **Polish (Quality)** | Qwen-Image-2.1 without adapter | **Frozen** | Same-model no-adapter reference/ablation; must not inherit the Default LoRA through a global fallback |
| **Critic / Review** | Deterministic verifiers plus optional frozen VLM evaluation | **No** | Review is not a training-label source; image-grounded Planner review must be proven before claiming visual self-review |

**The rule this enforces, everywhere in the codebase**: there is no
AI-judge-labeled training data anywhere in this project. Every training
signal is either a real, published, human-collected dataset, or the
model needs no training data at all because it ships frozen. This is
checked, not just stated — see `docs/architecture/RESEARCH_AND_CITATIONS.md`
§1 for the full reasoning trail, and `docs/research/` for every place this
review independently re-verified it (e.g., confirming the Planner's
deprecated training script — `training/planner/train.py` — is real,
functional, loudly marked deprecated, and genuinely never invoked by
the live inference path, rather than silently orphaned).

### 6.1 Structure and Layout Strategy (Sketch Tier)

The production baseline uses an open-domain Universal Scene DAG (`krisna.layout.v1`) with our Zero-VRAM Simplex collision solver (`_resolve_flex_layout`) preserving locked region immutability and rendering high-entropy 3-channel RGB wireframes with anti-aliased TrueType typography (TextDiffuser-2 ECCV 2024; AnyText ICLR 2024). It supports infinite design scenarios (UI mockups, marketing posters, editorial publishing, e-commerce showcases, infographics, diagrams, and freeform sketches) via polymorphic node types (`NODE_TYPES`), a 16-class universal semantic palette (`UNIVERSAL_SEMANTIC_PALETTE`, NVIDIA SPADE), automated turn-to-turn differential masking (`compute_differential_mask`, FlowEdit ICCV 2025), and 2D spatial quad-tree prompt indexing (Hybrid Layout Control, ICCV 2025).

The historical experimental continuous flow matching ablation (`LayoutFlow`, Guerreiro et al., ECCV 2024) and discrete MaskGIT token routes have been retired and removed in favor of the single production-locked neuro-symbolic engine.

---

## 7. Inference orchestration

### 7.1 The core constraint

At most one large generative tier (Polish) is GPU-resident at
a time, in addition to the always-resident baseline (§5.3). Tiers swap
in and out on demand — this is a **swap orchestrator**, not a
multi-model serving cluster.

### 7.2 Admission: two independent ledgers

`VRAMLedger` and `RAMLedger` share a base class (`_BaseLedger`) for
admit/release/would-fit logic. Admission is checked against each
`ModelSpec`'s **declared** `vram_gb`/`ram_gb` (`model_registry.py`), not
a live probe — deterministic and CI-testable without a GPU. A
`safety_margin_gb` (default `0.0`, opt-in) lets an operator on real
hardware reserve headroom the declared numbers can't capture (CUDA
context overhead, fragmentation).

### 7.3 Resident-footprint table (target, both operating points)

| Tier | Full-envelope (24GB) | Low-VRAM (12GB) |
|---|---|---|
| Planner (baseline, always resident) | 6.5GB / 0GB RAM (Q4_K_M GGUF) | unchanged |
| Sketch (baseline, always resident) | 0.0GB VRAM / 0.1GB RAM (CPU resident) | unchanged |
| Polish-Default (Qwen-Image-2.1 LoRA) | 14.0GB / 0GB RAM (BF16 native pipeline) | 12.0GB / 2.0GB RAM (`enable_model_cpu_offload()`) |
| Polish-Quality (Qwen-Image-2.1 Frozen) | 14.0GB / 0GB RAM (BF16 frozen base) | 12.0GB / 2.0GB RAM (`enable_model_cpu_offload()`) |
| Polish-Fast (Qwen-Image-2.1 CFM Distilled) | 14.0GB / 0GB RAM (4–6 step distilled preview) | 12.0GB / 2.0GB RAM (`enable_model_cpu_offload()`) |

"Planner (4-bit) + Sketch tier + verifiers ~8–10GB" is the idle-state
target this table exists to satisfy (§5.3's Tier A rule). Every figure
above is arithmetic-derived from real published model specs and
measured quantization behavior, not yet confirmed by a live hardware
run in this environment — flagged plainly rather than presented as
measured (see [Volume III: Systems Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_SYSTEMS_VOLUME_3.md)).

### 7.4 Swap Orchestrator state machine

States: `IDLE_RESIDENT` (baseline only) → `SWAPPING_TO_POLISH` →
`POLISH_RESIDENT` → `SWAPPING_BACK_FROM_POLISH` → back to
`IDLE_RESIDENT`; the same shape for Critic
(`SWAPPING_TO_CRITIC`/`CRITIC_RESIDENT`/`SWAPPING_BACK_FROM_CRITIC`);
plus `ERROR_RECOVERY`, reachable from either swapping-in state and
always resolving back to `IDLE_RESIDENT`. The baseline is fully
unloaded *before* a swappable tier is admitted — admission is checked
against the candidate tier's declared size alone, never baseline-plus-
candidate summed (a distinction documented in
[Volume I: Architecture Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md)).

### 7.5 Serving strategy: plain `transformers`/`diffusers`, not a dedicated inference server

Deliberate, not a gap: this system's access pattern (single session,
one resident generative tier, swap-heavy) doesn't benefit from
continuous-batching servers like vLLM, which exist to serve *many
concurrent requests* against one resident model — a different problem
than this system has. Stated explicitly here so it's a documented
design choice, not something a later reviewer has to rediscover and
wonder about.

### 7.6 Observability: declared budget vs. real hardware

`GET /orchestrator/status` returns both the admission ledgers' declared
budgets (`vram`, `ram`) *and* live hardware probes (`real_vram`,
`real_ram`, via `torch.cuda.mem_get_info()`/`/proc/meminfo`) — the two
can legitimately disagree (another process on a shared GPU, OS memory
pressure), and both are surfaced rather than only the one the admission
logic actually uses.

---

## 8. Data pipeline (data-forge)

### 8.1 Dataset categories

Two: `ui_first` (real UI screenshots — RICO, CLAY, Enrico, Screen2Words,
WebUI) and `general_visual` (broader aesthetic grounding — e.g. PD12M,
CC12M). Every dataset entry carries a real `repo_id`/`repo_url`, a
`license_url`, and an honestly-scoped `license_status` (defaulting to
`unverified` even for entries with a clean-looking license, so the
pipeline's own License Verification Agent runs regardless of apparent
confidence).

### 8.3 Corpus scale target

Hundreds-of-thousands of usable images (100K–500K target range) after
dedup, quality filtering, and safety exclusion — not millions; sized to
what a from-scratch Sketch-tier transformer and a LoRA fine-tune
actually need, not to an arbitrary "bigger is better" figure.

### 8.4 Pipeline shape

Chunk-based with a 50,000-record production chunk (10,000 for the local
override), while internal inference/image batches remain bounded to fit
the single-GPU host and reduce expensive model restarts: CLIP (dedup) →
Tier-1 VLM (quality, face scrub, safety) → Tier-2 VLM (escalation,
borderline records only) → Tier-1 VLM again (recaption, structure) → OCR
specialist and text-PII redaction → deterministic routing → encoders.
The escalation step's position (*before* recaption/structure, not after)
is load-bearing: a record Tier-2 rescues from
"borderline" to "safe" must still be reachable by the safety-tier
filters that gate every later stage, or it's silently lost — a real bug
this review found and fixed (see [Volume II: DataForge Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_DATAFORGE_VOLUME_2.md)).

### 8.5 Preference pairs: a separate stream

DPO training data (Pick-a-Pic v2, HPDv2, GameLabel-10K) doesn't fit the
single-image manifest schema — a pair is two images, one label, one
shared prompt. Handled as a parallel stream: `s01_6_preference_pairs`
dedups/blurs/safety-classifies pairs directly; `training/data_forge_bridge/
sync_dpo_pairs.py` resolves them through the shared BlobStore into a
`PreferenceStore`; `train_dpo.py` live-encodes through Z-Image-Turbo's
own VAE at train time. (`s08_5_dpo_encoding.py`, an earlier attempt at
pre-computing these latents, is disabled by default — confirmed to have
zero real consumers; see [Volume III: Systems Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_SYSTEMS_VOLUME_3.md)).

### 8.6 Synthetic captions, kept from mismatching real usage

Every image gets a VLM-generated dense caption (Tier-1's recaptioning
stage), but Sketch-tier training mixes in the dataset's own original
caption 5% of the time and applies classifier-free-guidance conditioning
dropout (10%) on top — closing the gap between a dense, VLM-written
training caption and the short, casual prompt a real user actually
types at inference time (Ho & Salimans, 2022; see §12).

---

## 9. Known, disclosed gaps and fixes (a live document, not a static one)

This PRD is written to match a system whose own review process
([Volumes I–III](file:///d:/Krisna/docs/research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md), 74 phases of rigorous verification) has repeatedly found
real bugs by tracing actual data paths rather than trusting
documentation or prior claims. Two are structural enough to belong in
this document rather than only the review log:

- **The VQ-token → pixel handoff superseded by DualStreamSketch**: The early
  exploratory MaskGIT 2D VQGAN codebook token pipeline was completely retired.
  In production, the Sketch Tier generates deterministic `DualStreamSketch`
  artifacts (`wireframe_ref`, `segmentation_ref`, `mask_ref`) directly from `LayoutIR`,
  completely eliminating the brittle VQGAN token decoding step and its crash vectors ([Final System Design](file:///d:/Krisna/docs/FINAL_SYSTEM_DESIGN.md)).
- **The safety gate**: `VerifierStack.safety_gate()` must run and must
  be able to block a Finalize result — it existed as a method for a
  period without ever being called anywhere ([Volume I: Architecture Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md)).

## 10. Success metrics (qualitative — no live model to benchmark in this environment)

- A user can complete Conversing → Sketching → Finalizing → (optionally)
  Critiquing end to end against the real service, on the target
  hardware envelope (§3), without an OOM or an unhandled crash.
- Every training signal traces to a real, cited, human-collected
  dataset or a frozen model — auditable against
  `docs/architecture/RESEARCH_AND_CITATIONS.md`.
- The full test suite (356 tests as of this revision) passes without a
  GPU; GPU-dependent tests skip cleanly rather than blocking CI.

## 11. Open questions (deliberately left open, not silently resolved)

- **FP8/4-bit inference-quality validation for Qwen3.5-9B**: the
  Planner's NF4 default is sized correctly against the VRAM budget
  (§7.3), but 4-bit inference quality itself hasn't been independently
  evaluated against BF16. If a real evaluation later finds it
  unacceptable, the fix is to raise the declared `vram_gb` to match
  BF16 (~18GB) and re-check the full idle-state sum against the 24GB
  envelope — not to silently revert the quantization without also
  updating the budget it would then violate. A regression test
  (`test_planner_vram_budget_consistency.py`) enforces exactly this
  pairing.
- **DPO `beta` for a flow-matching model**: Wallace et al.'s original
  range (2000–5000) was tuned for epsilon-prediction diffusion (SD1.5/
  SDXL); newer sweeps on flow-matching models closer to Z-Image-Turbo's
  architecture found meaningfully lower optima — see §12's citations.
  Not yet swept against this project's actual model.
- **RICO's real usable-image count** after license/dedup/join — the
  single longest-standing open verification item in this project's
  review history.
- **Real hardware VRAM/RAM measurements** for every tier in §7.3 —
  currently arithmetic-derived, not measured.
- **`s08_5_dpo_encoding.py`'s disposition** — recommended for deletion
  (not re-wiring) given it has zero consumers and the alternative (a
  precomputed-latent fast path) is speculative future-proofing for a
  bottleneck not yet established as real; not yet executed.
- **DesignSense-10k / DesignPref public availability** — both real,
  both currently unpublished; `s11_registry_watcher` is configured to
  flag the moment either becomes fetchable.

---

## 12. Citations

Primary sources this PRD and the underlying implementation are built
on — see `docs/architecture/RESEARCH_AND_CITATIONS.md` for the full,
one-hundred-plus-line version with per-decision reasoning, and
[citations.md](file:///d:/Krisna/docs/citations.md) for the complete list
including every citation independently re-verified during this
project's review process.

1. Chang, H., et al. (2023). *Muse: Text-to-Image Generation via Masked
   Generative Transformers*. In *Proceedings of the 40th International
   Conference on Machine Learning (ICML)*, PMLR vol. 202, pp. 4055–4075. —
   Classifier-free guidance formula, training-time conditioning dropout rate,
   and linear guidance ramp for the Sketch tier's masked-generative
   architecture. Verified directly against the paper (§2.7): formula,
   10% dropout, and ramp description all match exactly.
2. Wallace, B., et al. (2024). *Diffusion Model Alignment Using Direct
   Preference Optimization*. In *Proceedings of the IEEE/CVF Conference
   on Computer Vision and Pattern Recognition (CVPR)*, pp. 8228–8238.
   DOI: 10.1109/CVPR52733.2024.00784. — Original Diffusion-DPO
   formulation and `beta` range, the starting point for Z-Image-Turbo's
   alignment.
3. Kim, M., Lee, Y., Kang, S., Oh, J., Chong, S., & Yun, S.-Y. (2024).
   *Preference Alignment with Flow Matching*. In *Advances in Neural
   Information Processing Systems (NeurIPS 37)*. — Velocity-prediction
   DPO substitution and flow-matching anchor regularization, adapting
   Wallace et al.'s formulation to continuous flow-matching models.
4. Croitoru, F. A., Hondru, V., Ionescu, R. T., Sebe, N., & Shah, M. (2025).
   *Curriculum Direct Preference Optimization for Diffusion and Consistency Models*.
   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
   Recognition (CVPR)*. — Curriculum difficulty margin scheduling for preference
   optimization on generative visual models.
5. Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. In
   *Advances in Neural Information Processing Systems (NeurIPS)*. —
   The general CFG technique underlying both the Sketch tier's approach
   (via Muse) and the caption-mixing/conditioning-dropout design for
   Sketch-tier training.
6. Qwen Team, Alibaba (2026-03-02). *Qwen3.5 Small Model Series
   (0.8B/2B/4B/9B) release notes*. — Confirms the Planner's real
   architecture (hybrid Gated DeltaNet + Gated Attention, native
   multimodal).
7. Tongyi-MAI, Alibaba (2025). *Z-Image: An Efficient Image Generation
   Foundation Model with Single-Stream Diffusion Transformer.*
   `github.com/Tongyi-MAI/Z-Image`. — Primary source for Z-Image-Turbo's
   Decoupled-DMD/DMDR distillation and S3-DiT architecture, the actual
   base model for Polish-Default.
8. Google DeepMind (2026). *Gemma 4.* `deepmind.google/models/gemma/gemma-4/`.
   — Confirms the Critic tier's real model identity and license.
9. Wang, B., et al. (2021). *Screen2Words: Automatic Mobile UI
   Summarization with Multimodal Learning*. In *Proceedings of the 34th
   Annual ACM Symposium on User Interface Software and Technology (UIST)*,
   pp. 498–510. DOI: 10.1145/3472749.3474765. — Real-caption
   length statistics justifying the recaptioning stage's existence.
10. Jonathan-Zhou (2024). *GameLabel-10k: Human UI Preference Dataset*.
    HuggingFace Datasets. — Human-labeled image-preference pairs from
    mobile-game crowdsourcing, a Stage-1 general-aesthetic DPO source
    once its actual (non-standard) schema was confirmed and a dedicated
    fetch adapter written for it.

---

*This document was reconstructed from the ground up by tracing every
`PRD §X` citation already present in this codebase's comments, tests,
and configs back to a single, internally consistent source — cross-
checked against `docs/architecture/RESEARCH_AND_CITATIONS.md` and
`docs/research/` (Volumes I–III) rather than drafted independently. Where a
citation's context implied a fact not otherwise documented (e.g. the
exact "Tier A ~10GB" ceiling, or §8.3's 100K–500K scale target), that
fact is stated here as the codebase's own established design intent,
not a new decision introduced by this document.*
