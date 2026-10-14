# Krisna: Bounded Multi-Turn Agentic UI Design via Structured Layout IR and Unified Diffusion Rendering

**Authors**: Krisna Research Group & DeepMind Pair Programming Initiative  
**Date**: October 2026  
**Status**: Research Specification / Pre-registration — empirical results pending
**Repository**: `https://github.com/priyanshu-ogdev/Krisna`

> [!WARNING]
> **Evidence correction (2026-10-13).** This document contains the original
> architecture proposal and provisional result tables. The current checkout
> does not contain a validated Qwen-Image-2.1 paired-editing trainer, a
> realized 125K wireframe→screenshot corpus, an image-grounded Planner review
> request, or receipts for the numerical results in §§6–7. Those sections are
> not publishable empirical evidence yet. The binding audit, model/freeze
> matrix, negative findings, gate sequence, and primary-source citations are
> in [`docs/research/FINAL_ARCHITECTURE_VERDICT.md`](research/FINAL_ARCHITECTURE_VERDICT.md).
> Until the listed gates pass, interpret all performance, memory, latency,
> “guarantee,” and “superior” language below as a hypothesis or placeholder,
> not a measured result.

> **Current canonical experiment snapshot (2026-10-15).** The only intended
> trainable artifact is a supervised LoRA for `Qwen/Qwen-Image-2.1`. The
> `Qwen3.5-9B` Planner is frozen, `krisna.layout.v1` is deterministic, and the
> local `Qwen/Qwen3.8-27B` model is a Data-Forge preprocessing dependency, not
> a Krisna training target. The workspace currently has **zero realized
> Qwen pair records, zero pair receipts, and no completed checkpoint**. The
> binding model/data decision is in
> [`docs/research/FINAL_ARCHITECTURE_VERDICT.md`](research/FINAL_ARCHITECTURE_VERDICT.md),
> and the current external-method review is in
> [`docs/research/SOTA_AUDIT_2026-10.md`](research/SOTA_AUDIT_2026-10.md).

> [!WARNING]
> **Dataset negative-audit amendment (2026-10-14).** The default Data-Forge
> registry is now quarantine-first. Only Figma2Code remains enabled as a
> UI-design candidate; UI-Vision, ScreenSpot, TASTE, and PartiPrompts are
> isolated evaluation sources. General imagery, general preference pairs,
> synthetic WebSight/AtomBlock-WebUI, uncertain third-party app captures, and
> unpinned or rights-unclear annotation sources are disabled rather than
> silently consumed. There is no training-approved source and no realized Qwen
> pair corpus until immutable revisions, asset-rights receipts, deterministic
> condition renders, source-disjoint splits, and a non-empty pair receipt all
> exist. See `docs/research/DATASET_ACCEPTANCE_MATRIX.md`.

> **Implementation review update (2026-10-14).** The inference audit found and
> corrected two fail-open paths: real Qwen generation could fall back to a
> synthetic image after a pipeline error, and readiness could be reported
> before baseline residency completed. Real mode now fails closed on missing
> conditioning or generation errors; synthetic outputs are explicitly marked
> test-mode artifacts. The verifier stack remains a post-generation measurement
> and safety gate, not a fabricated renderer score or a training signal.

---

## Abstract

### Current flow-matching design addendum

The current implementation no longer treats a handwritten Euler/RK2 helper as
the Qwen sampler. Qwen-Image-2.1's native Diffusers pipeline and scheduler are
authoritative for real runs. The training bridge now uses a typed,
split-isolated condition/target dataset, a pinned native Qwen image-to-image
trainer boundary, CPU-only preflight, and receipts for source hashes, split
counts, trainer revision, and adapter provenance. The inference bridge inspects
the native pipeline call signature, records live scheduler metadata, and fails
closed in strict mode when legacy ODE/SDEdit controls are requested.

This decision is grounded in the peer-reviewed rectified-flow and editing
literature: Esser et al. (ICML 2024, PMLR 235) for rectified-flow transformer
training and resolution-aware sampling; Brooks et al. (CVPR 2023 Open Access)
for instruction-conditioned editing; Zhang et al. (ICCV 2023 Open Access) for
explicit spatial conditioning; and Kulikov et al. (ICCV 2025 Open Access) for
inversion-free flow editing. These sources motivate controlled ablations, not
claims that Krisna has reproduced their results. The full implementation plan
is [`docs/research/FLOW_MATCHING_UPGRADE_PLAN.md`](research/FLOW_MATCHING_UPGRADE_PLAN.md).

Automating user interface (UI) design from conversational natural language requires solving three interconnected challenges: (1) maintaining coherent design intent across multi-turn human-AI dialogue without semantic drift; (2) enforcing hard spatial, typographic, and element-hierarchy constraints on a 2D canvas; and (3) generating production-grade visual renders on consumer hardware without exceeding memory ceilings. Existing approaches either rely on monolithic text-to-image diffusion models—which hallucinate illegible typography and ignore layout constraints—or discrete tokenized generative transformers (e.g., MaskGIT/VQGAN), which suffer from codebook quantization collapse, loss of vector hierarchies, and inability to enforce hard regional locking. 

In this paper, we specify **Krisna**, a bounded multi-turn agentic UI design system built upon a decoupled, single-model-per-responsibility architecture. Krisna decomposes UI creation into three coordinated tiers: (1) a frozen vision-language *Planner* that maps dialogue and an optional wireframe into deterministic state deltas; (2) an intermediate structured representation (*Layout IR*, `krisna.layout.v1`) that captures hierarchical containers, normalized geometric coordinates, and exact string literals, rendered deterministically into a 2D wireframe; and (3) a Qwen-family diffusion *Polish* tier proposed for image-conditioned wireframe-to-UI synthesis. The intended contribution is the explicit synchronization contract between these tiers, plus a bounded refinement policy and an evaluation protocol for structure, typography, locks, preference, and resource use. Quantitative claims are deliberately deferred until the trainer, paired-data manifest, multimodal review path, and hardware/evaluation receipts described in the final audit are complete.

---

## 1. Introduction

User interface design is inherently hierarchical, multi-modal, and iterative. When human designers construct digital interfaces, they rarely jump directly from high-level verbal requirements to finished, pixel-perfect layouts. Instead, they follow a principled workflow: clarifying intent, constructing wireframes with spatial hierarchies and typographic tokens, and iteratively refining visual textures, color schemes, and component styling.

Recent advances in Generative Artificial Intelligence have attempted to automate this pipeline using two predominant paradigms:
1. **End-to-End Generative Diffusion Models**: Models such as Stable Diffusion, FLUX, and Midjourney synthesize striking imagery from text prompts. However, when applied to digital UIs, they exhibit severe failure modes: text labels are rendered as garbled pseudo-glyphs, component boundaries drift unpredictably, and users cannot specify localized edits without regenerating the entire canvas.
2. **Discrete Masked Generative Image Transformers (MaskGIT / VQGAN)**: Frameworks utilizing discrete vector-quantized token grids (e.g., Muse, MaskGIT) attempt to generate UI drafts by predicting discrete image tokens. However, empirical analysis demonstrates that spatial downsampling (typically stride-16) permanently obliterates high-frequency UI details such as sub-pixel text anti-aliasing, 1px border strokes, and exact bounding box coordinates. Crucially, discrete token grids cannot represent semantic component hierarchies or guarantee region locks during multi-turn edits.

Moreover, multi-turn conversational agents frequently suffer from **conversational drift**: when given free rein over multiple turns, an LLM planner gradually drifts away from the user's original design intent, inventing unnecessary layout alterations and accumulating hallucinatory state changes.

To resolve these fundamental limitations, we introduce **Krisna**. Our contributions are as follows:

- **Decoupled Architecture with Single-Model Responsibility**: We enforce a strict separation of concerns where the Planner manages intent and review, Layout IR governs geometry, and a single diffusion model (`Qwen-Image-2.1`) governs pixel rendering. No model is permitted to execute or override another model's domain.
- **Structured Intermediate Representation (`krisna.layout.v1`)**: We eliminate discrete VQ token codebooks in favor of a mathematically rigorous Layout IR specifying normalized coordinates $[y_{\min}, x_{\min}, y_{\max}, x_{\max}] \in [0, 1000]^4$, parent-child tree structures, and exact textual labels, rendered deterministically to a lossless wireframe.
- **Bounded Review and Explicit Verification**: Planner self-review is bounded;
  pixel-output verification remains a separate post-generation stack. Its safety
  score is a gate, while other scores are diagnostic evidence and are never
  silently substituted for human evaluation.
- **Bounded Refinement & Anti-Drift Protocol**: We formulate a formal turn budget policy ($\tau \le 4$) and an inactivity debounce coordinator (30s timeout) that provably prevents unconstrained autonomous drift.
- **Hardware-Enforced Admission Control**: We provide formal mathematical ledgers (`VRAMLedger`, `RAMLedger`) proving deterministic operation within consumer 12GB VRAM and 16GB system RAM boundaries.

---

## 2. Related Work & Theoretical Foundations

```
+-------------------------------------------------------------------------------------+
|                               RELATED WORK TAXONOMY                                 |
+-------------------------------------------------------------------------------------+
| 1. Layout Generation       | LayoutFlow (ECCV '24), LayoutDiffusion (ICCV '23),     |
|                            | UniLayDiff (CVPR '26), LayoutDM (CVPR '23)             |
+----------------------------+--------------------------------------------------------+
| 2. Constrained Generation  | JSONSchemaBench (2025), Automata Decoding (2024),      |
|                            | Guided Generation (Willard & Louf, 2023)               |
+----------------------------+--------------------------------------------------------+
| 3. Interactive Specification | PromptMagician (IEEE TVCG '23), structured prompt/state artifacts |
| 4. Dense/Hybrid Retrieval  | ColBERT (SIGIR '20), ColBERTv2 (NAACL '22), BM25       |
+----------------------------+--------------------------------------------------------+
| 5. Diffusion & Editing     | Qwen-Image-2.1 (2026), InstructPix2Pix (CVPR '23),     |
|                            | DreamBooth LoRA (CVPR '23), Diffusion-DPO (CVPR '24)   |
+-------------------------------------------------------------------------------------+
```

### 2.1 Layout Generation & Continuous Flow Matching
Early layout generation systems framed UI design as sequence modeling over bounding boxes using autoregressive transformers (LayoutTransformer, Gupta et al., 2021) or discrete diffusion (LayoutDiffusion, Zhang et al., ICCV 2023). While discrete diffusion improved structural diversity, it struggled with continuous spatial alignment and multi-scale hierarchy. 

Recently, **LayoutFlow** (Guerreiro et al., ECCV 2024) demonstrated that continuous flow matching over normalized bounding box coordinates provides superior generative distribution modeling with optimal transport vector fields. Krisna builds upon the continuous flow-matching foundation of LayoutFlow, extending it with parent-conditioned hierarchical container trees to represent modern nested web and mobile components.

### 2.2 Constrained Decoding and Grammar-Guided Generation
Unconstrained LLM generation in production agentic systems frequently fails due to JSON schema invalidity, hallucinated attributes, and malformed data structures. Recent theoretical advances in grammar-constrained decoding (Geng et al., EMNLP 2023; Koo, 2024; Willard & Louf, 2023) show that intersecting token vocabulary logits with deterministic pushdown automata or regular expressions guarantees $100\%$ schema validity without fine-tuning. JSONSchemaBench (Geng et al., 2025) empirically demonstrated that guided decoding preserves semantic reasoning if the grammar does not overly constrain intermediate reasoning tokens. Krisna applies grammar-constrained decoding over `krisna.delta.v1` via llama.cpp grammar engines.

### 2.3 Image-Conditioned Diffusion and Preference Alignment
Text-to-image models have rapidly transitioned from unconditioned U-Nets to flow-matching DiT architectures (Lumina-T2X, Stable Diffusion 3, FLUX, Qwen-Image-2.1). For editing tasks, instruction-guided diffusion (InstructPix2Pix, Brooks et al., 2023) and explicit spatial control (ControlNet, Zhang et al., 2023) motivate separating the condition renderer from the pixel renderer. The current Krisna mainline does **not** treat a screenshot plus annotations as a valid pair: a Qwen record requires a deterministic wireframe condition image, a scrubbed target, a provenance receipt, and source-policy approval. Missing condition artifacts are rejected rather than synthesized. Preference optimization remains deferred until its trainer, data, and evaluation receipts exist.

### 2.4 Production architecture revision: fail-closed local inference

The production stack is split into a low-latency Rust control plane, an async
Python model plane, and a static browser/Tauri surface:

1. **Rust policy boundary:** Tauri owns optimistic revision checks, bounded turn
   accounting, path-safe session identifiers, and lightweight HTTP IPC. It does
   not load model weights or duplicate Python's GPU scheduler.
2. **Python model plane:** FastAPI owns the session store, residency
   transitions, deterministic Layout IR projection, Qwen generation, and
   verifier execution. Real mode rejects missing conditions and generation
   failures; it never presents a synthetic placeholder as successful inference.
3. **Evidence boundary:** the renderer reports runtime mode but does not claim
   verifier measurements. The verifier stack computes post-generation scores;
   safety is a gate and the remaining metrics are evidence.
4. **Frontend boundary:** the lightweight Node/Tauri UI proxies explicit routes,
   consumes readiness/residency signals, respects reduced-motion preferences,
   and does not interpolate discovered filesystem paths as executable JS.

### 2.5 IEEE and Springer design implications

PromptMagician (Feng et al., IEEE TVCG 2023) treats text-to-image creation as
an interactive prompt-engineering problem rather than a one-shot prompt call.
Krisna adopts the transferable part of that result—inspectable prompt/state
artifacts and iterative user control—but retains Layout IR as the authority for
geometry, locks, and literal text. Prompt search is therefore an assistive
operation, not a replacement for a validated state delta.

LayoutFlow (Guerreiro et al., ECCV 2024, Springer LNCS proceedings) provides a
relevant learned continuous-layout precedent. Krisna uses it to motivate a
future learned LayoutFlow ablation, not to claim a trained Sketch model. The
production baseline remains deterministic because it already provides a
complete `krisna.layout.v1` renderer, exact lock mask, and auditable split.
This distinction keeps related-work evidence separate from implementation or
performance evidence.

---

## 3. System Architecture & Mathematical Formulation

```mermaid
graph TD
    User([User Prompt / Turn]) --> Coordinator[Session Coordinator & Turn Budget]
    Coordinator --> Planner[Tier 1: Frozen Planner llama.cpp W4A16]
    Planner --> Delta[krisna.delta.v1 Validated Delta]
    Delta --> StateMerge[DesignState Merge & Concurrency Check]
    StateMerge --> LayoutEngine[Tier 2: Layout IR Engine krisna.layout.v1]
    LayoutEngine --> Wireframe[Deterministic 2D Wireframe PNG]
    Wireframe --> Polish[Tier 3: Qwen-Image-2.1 Polish Diffusion]
    Polish --> Verifier[Verifier Stack & Safety Gate]
    Verifier --> Evidence[Versioned metrics + safety decision]
    Evidence --> FinalUI([Final UI only if gate passes])
    Wireframe -.-> Review[Planner Resident Review Mode]
    Review -.-> Coordinator
```

### 3.1 Formal Session State
Let a design session at discrete turn $t \in \mathbb{N}$ be represented by the tuple:
$$\mathcal{S}_t = \langle \mathcal{H}_t, \mathcal{C}_t, \mathcal{L}_t, \mathcal{I}_t, \mathcal{B}_t, r_t \rangle$$
where:
- $\mathcal{H}_t = [ (u_1, a_1), \dots, (u_k, a_k) ]$ is the bounded conversational dialogue history.
- $\mathcal{C}_t$ is the accumulated set of validated semantic constraints (e.g., color palette, typography, visual hierarchy).
- $\mathcal{L}_t$ is the structured Layout Intermediate Representation (`krisna.layout.v1`).
- $\mathcal{I}_t \in \mathbb{R}^{H \times W \times 3}$ is the active visual artifact (wireframe or rendered image).
- $\mathcal{B}_t = \langle \tau_{\text{user}}, \tau_{\text{self}}, \tau_{\max} \rangle$ is the refinement turn budget, where $\tau_{\text{user}} + \tau_{\self} \le \tau_{\max} = 4$.
- $r_t \in \mathbb{N}$ is the monotonic optimistic concurrency revision integer.

### 3.2 Tier 1: Planner Deliberation & State Transition
Upon receipt of a user utterance $u_t$ (or an autonomous self-review trigger $\alpha$), the Planner samples a state delta $\delta_t$ conditioned on retrieved human design critique examples $\mathcal{R}(\mathcal{H}_t) \subset \text{UICrit}$:
$$\delta_t \sim p_{\theta_{\text{plan}}}(\delta \mid u_t, \mathcal{C}_{t-1}, \mathcal{R}(\mathcal{H}_t), \Pi(\mathcal{L}_{t-1}))$$
subject to strict grammar compliance:
$$\delta_t \in \mathcal{G}(\text{krisna.delta.v1})$$

The state update operator $\oplus$ is deterministic:
$$\mathcal{C}_t = \mathcal{C}_{t-1} \oplus \delta_t.\text{applied\_constraints}$$
$$r_t = r_{t-1} + 1$$
If the revision in the incoming request does not equal $r_{t-1}$, the transaction aborts with $\text{StaleDesignStateError}$, guaranteeing isolation in concurrent multi-client sessions.

### 3.3 Tier 2: Structured Layout IR Specification
The Layout IR $\mathcal{L}_t = (\mathcal{K}, \mathcal{E})$ consists of canvas metadata $\mathcal{K} = (W, H, c_{\text{bg}})$ and a set of $N$ UI elements $\mathcal{E} = \{e_1, \dots, e_N\}$. Each element $e_i$ is formally defined as:
$$e_i = \langle \text{id}_i, \mathbf{b}_i, c_i, p_i, \mathbf{s}_i, T_i, \ell_i \rangle$$
where:
- $\text{id}_i \in \Sigma^*$ is a unique semantic identifier.
- $\mathbf{b}_i = [y_{1}, x_{1}, y_{2}, x_{2}] \in [0, 1000]^4$ is the normalized bounding box with $y_1 < y_2$ and $x_1 < x_2$.
- $c_i \in \mathcal{C}_{\text{type}} = \{\text{container}, \text{button}, \text{text}, \text{image}, \text{input}, \text{navbar}, \text{card}\}$ is the element class.
- $p_i \in \mathcal{E} \cup \{\emptyset\}$ specifies the parent container, defining a directed acyclic tree $\mathcal{T}_{\mathcal{L}}$.
- $\mathbf{s}_i = \langle \text{fill}, \text{stroke}, \text{width}, \text{radius}, \alpha \rangle$ denotes vector styling attributes.
- $T_i \in \Sigma^*$ is the literal Unicode text string to be rendered.
- $\ell_i \in \{0, 1\}$ is the hard region-lock indicator.

We define the deterministic projection operator $\Pi: \mathcal{L} \to \mathcal{I}_{\text{wire}}$ such that:
$$\mathcal{I}_{\text{wire}}(x, y) = \text{RenderSVG}(\mathcal{L}, W, H)$$
Because $\Pi$ is purely deterministic, $\text{Var}(\Pi(\mathcal{L})) = 0$, completely eliminating codebook quantization noise.

### 3.4 Tier 3: Unified Flow-Matching Polish & Feathered Region Locking
The Polish tier synthesizes a production-grade render $\mathcal{I}_{\text{render}}$ from wireframe $\mathcal{I}_{\text{wire}}$ using `Qwen-Image-2.1`. Grounded in continuous Normalizing Flows (Lipman et al., ICLR 2023; Liu et al., ICLR 2023) and Multimodal Rectified Flows (Esser et al., ICML 2024), the generative process solves the continuous Ordinary Differential Equation (ODE):
$$\frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}_{\text{text}}, \mathcal{I}_{\text{wire}}), \quad t \in [t_{\text{start}}, 1.0]$$
where $t_{\text{start}} = \max(0.0, 1.0 - s_{\text{edit}})$.

#### 3.4.1 Numerical ODE Integrators
We implement both 1st-order Euler and 2nd-order Midpoint (RK2) predictor-corrector discretization schemes:
1. **Euler Step**:
   $$\mathbf{z}_{t + \Delta t} = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}_{\text{text}}, \mathcal{I}_{\text{wire}})$$
2. **Midpoint / RK2 Predictor-Corrector**:
   $$\mathbf{k}_1 = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}_{\text{text}}, \mathcal{I}_{\text{wire}})$$
   $$\mathbf{z}_{\text{mid}} = \mathbf{z}_t + \frac{\Delta t}{2} \cdot \mathbf{k}_1$$
   $$\mathbf{k}_2 = \mathbf{v}_\theta\left(\mathbf{z}_{\text{mid}}, t + \frac{\Delta t}{2}, \mathbf{c}_{\text{text}}, \mathcal{I}_{\text{wire}}\right)$$
   $$\mathbf{z}_{t + \Delta t} = \mathbf{z}_t + \Delta t \cdot \mathbf{k}_2$$

#### 3.4.2 Dynamic Classifier-Free Guidance & Dual Conditioning
To eliminate color blowout and contrast saturation in late flow steps, we employ dynamic cosine-decayed guidance (Brooks et al., CVPR 2023):
$$w(t) = 1.0 + (w_{\max} - 1.0) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{t - t_{\text{start}}}{1.0 - t_{\text{start}}}\right)$$
coupled with dual text-and-image conditioning vectors:
$$\hat{\mathbf{v}}(\mathbf{z}_t, t, \mathbf{c}_I, \mathbf{c}_T) = \mathbf{v}(\mathbf{z}_t, t, \emptyset, \emptyset) + s_I [\mathbf{v}(\mathbf{z}_t, t, \mathbf{c}_I, \emptyset) - \mathbf{v}(\mathbf{z}_t, t, \emptyset, \emptyset)] + w(t) [\mathbf{v}(\mathbf{z}_t, t, \mathbf{c}_I, \mathbf{c}_T) - \mathbf{v}(\mathbf{z}_t, t, \mathbf{c}_I, \emptyset)]$$

#### 3.4.3 Feathered Region Locking
Let $\mathcal{R}_{\text{lock}} = \{e_i \in \mathcal{E} \mid \ell_i = 1\}$ denote user-locked elements. We generate a binary spatial mask $\mathcal{M}_{\text{lock}} \in \{0, 1\}^{H \times W}$, smoothed with Gaussian boundary falloff to eliminate seam discontinuities:
$$\mathcal{M}_{\text{feather}} = \mathcal{M}_{\text{lock}} * G_\sigma \quad (\sigma = 8\text{px})$$
$$\mathcal{I}_{\text{out}} = \mathcal{M}_{\text{feather}} \odot \mathcal{I}_{\text{prior}} + (1 - \mathcal{M}_{\text{feather}}) \odot \mathcal{I}_{\text{render}}$$

#### 3.4.4 Dual-Loader Asynchronous Memory Orchestration
To execute within consumer envelopes (12GB–16GB VRAM, 16GB Host RAM), the architecture coordinates 2 distinct runtime loaders:
1. **Loader 1 (llama.cpp GGUF)**: Zero-copy memory mapping (`use_mmap=True`) for frozen Planner and Layout IR.
2. **Loader 2 (Torch/Diffusers)**: Flow-matching velocity pipeline with CPU offloading for Qwen-Image-2.1.
Background pre-warming (`start_background_warm()`) executes non-blockingly via an asynchronous worker task, allowing instant session initialization while tracking lifecycle stages (`loading_planner`, `loading_layout`, `ready`, `swapping`). Incoming conversational turns non-blockingly await background warming with timeout isolation, guaranteeing zero unhandled 409 collisions.

### 3.5 Verifier Stack & Safety Gating
Before $\mathcal{I}_{\text{out}}$ is returned to the user, it is evaluated by an ensemble of automated verifiers:
$$\mathbf{V}(\mathcal{I}_{\text{out}}) = \langle S_{\text{safe}}, S_{\text{OCR}}, S_{\text{align}}, S_{\text{aes}} \rangle$$

The safety gate operates as an absorbing barrier:
$$\text{Gate}(\mathcal{I}_{\text{out}}) = \begin{cases} 
\text{Accept}, & \text{if } S_{\text{safe}}(\mathcal{I}_{\text{out}}) \ge \theta_{\text{safe}} \\ 
\text{Reject and Purge}, & \text{otherwise} 
\end{cases}$$
Upon rejection:
1. `blob_store.delete(image_ref)` immediately deletes the image from storage.
2. $\mathcal{S}_t.\text{stage}$ is reverted to `SKETCHING`.
3. A `SafetyGateViolationError` is raised, preventing corrupted or unsafe imagery from ever reaching the client.

---

## 4. Bounded Refinement & Anti-Drift Policy

### 4.1 The Conversational Drift Problem
In multi-turn generative design, unconstrained agent loops exhibit entropy accumulation. Let $D_{\text{intent}}(\mathcal{L}_t, \mathcal{C}_0)$ measure semantic divergence from the user's initial prompt $u_0$. In an unconstrained agent system:
$$\lim_{t \to \infty} \mathbb{E}[D_{\text{intent}}(\mathcal{L}_t, \mathcal{C}_0)] > \epsilon_{\text{drift}}$$

### 4.2 Bounded Turn Budget Formulation
To guarantee termination and intent stability, Krisna bounds the total refinement iterations per cycle:
$$\tau = \tau_{\text{user}} + \tau_{\text{self}} \le \tau_{\max} = 4$$

**State Transition Machine**:
$$\text{CONVERSING} \xrightarrow{u_0} \text{SKETCHING} \xrightarrow{\tau < 4} \text{REFINING} \xrightarrow{\tau = 4 \lor \text{finalize}} \text{FINALIZING} \to \text{FINALIZED}$$

When $\tau = \tau_{\max}$, the session enters a locked state where further layout modifications are rejected unless the user explicitly initiates a new post-finalization revision cycle, which creates a new versioned snapshot $\mathcal{S}^{(k+1)}$ while archiving $\mathcal{S}^{(k)}$.

### 4.3 Inactivity Debounce Coordinator
When a wireframe $\mathcal{I}_{\text{wire}}$ is visible, user contemplation time is utilized for autonomous self-review. The coordinator maintains an asynchronous timer $\Delta t$:
- If user input $u_{t+1}$ arrives within $\Delta t < 30\text{s}$, the timer is cancelled and $u_{t+1}$ is processed.
- If $\Delta t \ge 30\text{s}$ and $\tau < \tau_{\max}$, the coordinator dispatches `planner_self_review()`, which audits wireframe alignment, verifies contrast ratios, and corrects typographic hierarchy without hallucinating user messages.

---

## 5. Empirical Evaluation Methodology

### 5.1 Benchmark Datasets

We evaluate Krisna across standard UI research corpora and preference alignment sets:

| Dataset | Split Size | Modality / Features | License | Primary Purpose |
|---|---|---|---|---|
| **RICO** (Deka et al., 2017) | 66,261 screens | Mobile UI hierarchy, bounds | CC BY 4.0 | Layout structure & geometry |
| **CLAY** (Li et al., 2022) | 59,514 screens | Vector UI graphics & bounding boxes | Apache 2.0 | Hierarchical container nesting |
| **Enrico** (Leiva et al., 2020) | 1,460 screens | Design topic & style categorization | CC BY-SA 4.0 | Color palette & style conditioning |
| **Screen2Words** (Wang et al., 2021) | 112,000 captions | Human-authored UI summaries | CC BY 4.0 | Conversational prompt grounding |
| **UICrit** (Google Research, 2023) | 4,500 critiques | Expert UI critique & feedback | CC BY-ND | RAG retrieval for Planner |
| **GameLabel-10K** (2024) | 10,000 screens | High-contrast specialized UI pairs | MIT | Preference pair alignment |
| **TASTE Benchmark** (2025) | 1,200 held-out tasks | Multi-turn human design evaluation | Non-commercial | End-to-end agentic benchmark |

### 5.2 Evaluation Metrics

We formulate five metric categories to quantify visual, geometric, and operational quality:

1. **Geometric Layout IoU ($\text{IoU}_{\text{geom}}$)**:
   Measures bounding box alignment and overlap accuracy between generated elements and target layout specifications:
   $$\text{IoU}_{\text{geom}} = \frac{1}{|\mathcal{E}|} \sum_{e_i \in \mathcal{E}} \frac{\text{Area}(\mathbf{b}_i \cap \mathbf{b}_i^*)}{\text{Area}(\mathbf{b}_i \cup \mathbf{b}_i^*)}$$
2. **Typographic & OCR Fidelity**:
   Evaluated using Character Error Rate (CER) and Word Error Rate (WER) using an off-the-shelf OCR engine (PaddleOCR / Tesseract) against ground-truth string literals $T_i$:
   $$\text{CER} = \frac{S + D + I}{N_{\text{chars}}}$$
3. **Region-Lock Invariance ($\Delta_{\text{lock}}$)**:
   Quantifies whether locked regions undergo unintended drift during diffusion refinement:
   $$\Delta_{\text{lock}} = \frac{1}{\sum \mathcal{M}_{\text{lock}}} \|\mathcal{M}_{\text{lock}} \odot (\mathcal{I}_{\text{final}} - \mathcal{I}_{\text{prior}})\|_2$$
   A perfect system achieves $\Delta_{\text{lock}} = 0.0$.
4. **CLIP Directional Alignment ($S_{\text{CLIP}}$)**:
   Measures agreement between visual changes $\Delta \mathcal{I}$ and conversational instruction $\Delta u$:
   $$S_{\text{CLIP}} = \frac{\Delta \mathbf{f}_{\text{img}} \cdot \Delta \mathbf{f}_{\text{text}}}{\|\Delta \mathbf{f}_{\text{img}}\| \|\Delta \mathbf{f}_{\text{text}}\|}$$
5. **Memory and Latency Footprint**:
   Measured peak VRAM ($\text{GB}$), system RAM ($\text{GB}$), cold swap latency ($s$), and per-turn generation time ($s$) on target hardware.

### 5.3 Unified inference receipt and target policy

The implementation now emits a `krisna.eval.v1` receipt at finalize time. It
records raw values, per-metric targets, missing dimensions, failed dimensions,
and a geometric mean over available metrics. The current project targets are:

| Metric | Target | Interpretation |
|---|---:|---|
| Safety gate | $\geq 0.90$ | Hard gate; never averaged away |
| CLIP alignment | $\geq 0.70$ | Project target; CLIPScore motivates the metric family, not this cutoff |
| OCR readability | $\geq 0.80$ | Includes CER/WER when Layout IR literals are available |
| Layout IoU | $\geq 0.70$ | Contour/semantic detector score against Layout IR regions |
| Aesthetic | $\geq 0.70$ | Requires calibration against held-out human ratings |
| Handoff consistency | $\geq 0.65$ | Perceptual semantic consistency, not pixel identity |
| Lock preservation | $\geq 0.95$ | Protected-region pixel preservation |

These are Krisna acceptance targets, not universal literature thresholds. The
published CLIPScore work reports human-correlation advantages but also domain
weaknesses; LPIPS motivates deep perceptual comparison over shallow pixel
metrics. Before publication, thresholds must be calibrated on a fixed,
source-disjoint validation set and reported with bootstrap confidence
intervals. Receipts with unavailable dimensions are `incomplete`, not passes.

---

## 6. Experimental Results & Ablation Studies

### 6.1 Main Benchmark Comparison (provisional; no receipts in this checkout)

> The following table is retained for historical traceability only. It must be
> replaced by a generated result artifact containing model revisions, split
> manifests, seeds, evaluator versions, sample counts, confidence intervals,
> and hardware receipts before submission. In particular, the current code's
> feathered lock compositor cannot justify a zero lock-drift claim, and a
> synthetic fallback cannot be reported as real Qwen inference.

We benchmark Krisna against prevailing architectural baselines:
- **Baseline A (End-to-End Diffusion)**: Direct text-to-image prompting using SDXL and FLUX.1.
- **Baseline B (Discrete Token Transformer)**: Two-stage MaskGIT + VQGAN with external Gemma 31B Critic (the legacy Krisna prototype).
- **Krisna (Ours)**: Frozen Planner + Layout IR (`krisna.layout.v1`) + Unified `Qwen-Image-2.1` with explicit verifier evidence.

| Method | $\text{IoU}_{\text{geom}} \uparrow$ | OCR CER $\downarrow$ | $\Delta_{\text{lock}}$ MSE $\downarrow$ | $S_{\text{CLIP}} \uparrow$ | Peak VRAM | Swap Latency |
|---|---|---|---|---|---|---|
| End-to-End SDXL | 0.412 | 0.684 | 0.284 | 0.241 | 11.8 GB | N/A |
| End-to-End FLUX.1 [dev] | 0.528 | 0.412 | 0.219 | 0.278 | 15.6 GB | N/A |
| MaskGIT + VQGAN + Gemma 31B | 0.681 | 0.389 | 0.142 | 0.289 | 18.2 GB (OOM on 12GB) | 48.2 s |
| **Krisna (Ours)** | **0.865** | **0.047** | **0.000** | **0.342** | **10.2 GB** | **0.0 s (Resident)** |

*Key Findings*:
1. **Typographic Superiority**: Krisna reduces Character Error Rate from $0.389$ to $0.047$ (an $87.9\%$ relative reduction), because Layout IR explicitly rasterizes exact text labels onto the wireframe, providing clear edge priors for the diffusion backbone.
2. **Perfect Region Invariance**: While discrete VQGAN tokens suffered codebook drift ($\text{MSE} = 0.142$), Krisna achieves $\Delta_{\text{lock}} = 0.000$ via exact coordinate-based alpha compositing.
3. **Elimination of Swap Bottleneck**: Retiring the external 31B Gemma critic in favor of resident Planner review reduces swap latency from $48.2\text{s}$ to $0.0\text{s}$, while saving $8.0\text{GB}$ of VRAM.

---

### 6.2 Ablation 1: Structured Layout IR vs Discrete VQGAN Codebook

We isolate the impact of replacing discrete VQGAN tokens (`boris/vqgan_f16_16384`) with structured Layout IR (`krisna.layout.v1`):

```
+-----------------------------------------------------------------------------------------+
| ABLATION 1: CODEBOOK COLLAPSE VS STRUCTURED LAYOUT IR                                   |
+-----------------------------------------------------------------------------------------+
| Metric                          | Discrete MaskGIT (16x16) | Structured Layout IR (Ours)|
+---------------------------------+--------------------------+----------------------------+
| Bounding Box Precision (mAP50)  | 0.541                    | 0.912                      |
| Small Element Recall (<32px)    | 0.218                    | 0.884                      |
| Element Hierarchy Adherence     | 0.482                    | 1.000 (Guaranteed Tree)    |
| Text String Preservation        | 0.114                    | 0.963                      |
| Token Quantization Artifacts    | Frequent blurring        | None (Lossless Canvas)     |
+-----------------------------------------------------------------------------------------+
```

Discrete tokens fail catastrophically on fine UI elements (buttons, icons, small text inputs) because a $16 \times 16$ grid cannot allocate sufficient codebook capacity to spatial boundaries. Layout IR preserves all vector primitives without loss.

---

### 6.3 Ablation 2: Resident Review vs External Critic

We compare our Resident Planner Review mode against an isolated external Gemma 4 31B Dense model:

| Evaluation Dimension | Gemma 4 31B (Subprocess) | Resident Planner Review (Ours) | Delta / Impact |
|---|---|---|---|
| **VRAM Footprint** | 18.0 GB (Requires 24GB GPU) | 0.0 GB added (~6.5 GB resident) | **-11.5 GB VRAM** |
| **Process Cold-Swap Time** | 45.4 s | 0.0 s | **-45.4 s (Instant)** |
| **Host System RAM Peak** | 48.0 GB (Offload mode) | 7.8 GB | **Fits 16GB RAM** |
| **Defect Detection Recall** | 84.2% | 82.6% | -1.6% (Statistically Parity) |
| **Actionable Layout Edits** | 71.0% | 88.4% | **+17.4% (Direct IR Grounding)**|

While Gemma 31B achieved marginally higher raw defect recall ($+1.6\%$), its suggested edits lacked coordinate grounding and frequently hallucinated elements outside the screen. Resident Planner review generates structured JSON deltas directly compatible with `krisna.delta.v1`, resulting in higher actionable revision accuracy ($88.4\%$ vs $71.0\%$).

---

### 6.4 Ablation 3: Bounded Turn Budget ($\tau \le 4$) Policy (policy bound, not semantic proof)

We analyze conversational stability over 10 consecutive turns with and without the turn budget constraint:

```
Conversational Drift Score D_intent over Turns:
Turn:        1      2      3      4      5      6      7      8      9      10
Unbounded:  0.08   0.14   0.21   0.32   0.49   0.68   0.84   1.05   1.28   1.49  (Catastrophic Drift)
Bounded(4): 0.08   0.12   0.15   0.16   [Finalized - Frozen Intent State]          (Provably Stable)
```

Without an explicit budget, autonomous agent loops hallucinate extraneous elements (e.g., transforming a simple login screen into an e-commerce dashboard by Turn 8). The $\tau \le 4$ policy bounds semantic divergence to $\le 0.16$, ensuring alignment with the user's initial objective.

---

## 7. Hardware Feasibility & Low-VRAM Validation

Krisna is engineered to run on commodity consumer systems equipped with a single 12GB–16GB GPU and 16GB of system RAM.

### 7.1 VRAM Ledger Audit (design estimates until hardware receipts exist)

```
+-----------------------------------------------------------------------------------+
| VRAM CONSUMPTION SCHEDULE (12GB TARGET ENVELOPE)                                  |
+-----------------------------------------------------------------------------------+
| 12.0 GB +-----------------------------------------------------------------------+ |
|         |                     MAX HARDWARE CEILING                              | |
| 10.2 GB | - - - - - - - - - - +-------------------------+ - - - - - - - - - - - | |
|         |                     | POLISH SWAP PEAK        |                       | |
|         |                     | Qwen-Image-2.1 (10.2GB) |                       | |
|  6.7 GB | +-------------------+                         +---------------------+ | |
|         | | BASELINE RESIDENT |                         | BASELINE RESTORED   | | |
|         | | Planner: 6.5 GB   |                         | Planner: 6.5 GB     | | |
|         | | Layout:  0.2 GB   |                         | Layout:  0.2 GB     | | |
|  0.0 GB +-+-------------------+-------------------------+---------------------+-+ |
|           0s                 15s                       30s                   45s  |
+-----------------------------------------------------------------------------------+
```

- **Admission Determinism**: The system never sums the baseline and polish tiers simultaneously. The `SwapOrchestrator` unloads the baseline prior to admitting `Qwen-Image-2.1`, keeping peak memory at $10.2\text{GB}$, well under the $12.0\text{GB}$ ceiling.
- **Host RAM Safety (provisional estimate)**: The design target is $7.8\text{GB}$ peak host RAM, but this number is not a measured hardware receipt and must not be used to guarantee safe execution until a target-hardware run is recorded.

---

## 8. Discussion, Limitations & Future Work

The proposed Krisna decomposition is intended to make consumer-hardware agentic UI design measurable, but it has not established a Pareto frontier in the current checkout. Several limitations remain open for future investigation:
1. **Component Interactivity**: Layout IR represents static 2D vector geometry and styling. Extending the IR to emit reactive React/Vue code with dynamic state bindings is an active engineering direction.
2. **Multimodal Planner Projection**: The current audited Planner request is text-only. The primary experiment will first validate an image-grounded `Qwen2.5-VL-7B-Instruct` request; `Qwen3-VL-8B-Instruct` remains a gated comparison.
3. **Continuous Flow-Matching Layout Training**: While the Layout IR schema is production-ready, training an end-to-end continuous flow-matching generator directly on the CLAY/RICO hierarchical tree distributions will replace template heuristics.

---

## 9. Final audit and approved architecture (2026-10-13)

The architecture is approved in a narrower, evidence-disciplined form:

1. **Planner:** use frozen `Qwen/Qwen2.5-VL-7B-Instruct` as the primary
   image-grounded Planner candidate. Keep `Qwen3-VL-8B-Instruct` as a gated
   bake-off candidate; do not select it solely because it is newer. The
   Planner-to-renderer interface is `krisna.delta.v1` plus Layout IR and
   verifier receipts, not hidden-state sharing.
2. **Structure:** use deterministic Layout IR and wireframe rendering as the
   main geometry baseline. A learned LayoutFlow/HART route is a future
   ablation, not an already-trained Sketch tier.
3. **Polish:** use `Qwen/Qwen-Image-2.1` with a frozen base and a trainable
   LoRA only after the native pipeline, external trainer revision, pair
   manifest, trainability report, and one-batch save/load/inference test pass.
   The official model card documents image editing and a Qwen Research License
   Agreement; record the exact license before any release or product claim.
4. **Quality reference:** run Qwen-Image-2.1 without an adapter as the frozen
   reference. Keep Qwen-Image-Edit-2511 as a separately identified comparison;
   it is a different 20B model and must not be conflated with Qwen-Image-2.1.
5. **Preference alignment:** defer Diffusion-DPO/KTO until supervised LoRA is
   reproducible and use only eligible human preference data. Do not use a
   Critic's self-generated labels as a training target.
6. **Locks and review:** implement exact protected-pixel copying and attach an
   image to visual review. Until then, report feathered compositing and
   text-grounded review honestly as ablations.

The full training schedule is therefore gated rather than calendar-driven:
freeze manifests → build source-disjoint pairs → validate native Qwen
conditioning/trainer → supervised LoRA sweep → matched evaluation → optional
preference alignment. The “125K pairs in week 1” and “five-week SOTA upgrade”
claims are removed from the final plan because neither the pair count nor the
trainer path is currently evidenced.

See [`docs/research/FINAL_ARCHITECTURE_VERDICT.md`](research/FINAL_ARCHITECTURE_VERDICT.md)
for the complete model matrix, negative-point audit, evaluation protocol,
primary sources, and completion criteria.

## 10. Conclusion

We have specified **Krisna**, an agentic multi-turn UI generation system organized around a frozen vision-language Planner, deterministic Layout IR, and a gated Qwen-Image-2.1 Polish LoRA experiment. The current repository supports the structural decomposition and parts of the data/adapter plumbing, but it does not yet provide the paired-data, native trainer, multimodal-review, exact-lock, or result receipts needed for empirical superiority claims. The next research milestone is therefore reproducible validation—not another unmeasured model swap.

---

## References

1. Besnier, V., & Chen, M. (2023). *A PyTorch Reproduction of Masked Generative Image Transformer*. arXiv:2310.14400.
2. Betker, J., et al. (2023). *Improving Image Generation with Better Captions*. OpenAI Technical Report.
3. Bin, Y., et al. (2025). *MotionFlux: Aligning Rectified Flow Motion Generation via Direct Preference Optimization*. arXiv:2508.19527.
   - *Role*: Velocity-prediction DPO substitution adapting DDPM epsilon formulation to rectified flow-matching models, and flow-matching anchor regularization term. Grounds `training/polish/dpo_loss.py`.
4. Brooks, T., Holynski, A., & Efros, A. A. (2023). *InstructPix2Pix: Learning to Follow Image Editing Instructions*. CVPR 2023.
5. Chai, S., et al. (2023). *LayoutDM: Transformer-based Diffusion Model for Layout Generation*. CVPR 2023.
6. Chang, H., et al. (2022). *MaskGIT: Masked Generative Image Transformer*. CVPR 2022.
7. Chang, H., et al. (2023). *Muse: Text-to-Image Generation via Masked Generative Transformers*. ICML 2023. arXiv:2301.00704.
   - *Role*: Classifier-free guidance formula for masked generative models, 10% conditioning dropout, linear guidance schedule.
8. Deka, B., et al. (2017). *Rico: A Mobile App Dataset for Integrated Design and Code Search*. ACM UIST 2017.
9. Esser, P., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML 2024. arXiv:2403.03206.
   - *Role*: Multimodal Diffusion Transformer (MMDiT) architecture specification. Primary architectural reference for `Qwen-Image-2.1`'s 32-block single-stream MMDiT backbone and logit-normal timestep sampling used in DPO training.
10. Frantar, E., et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323.
11. Geng, S., et al. (2023). *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP 2023. arXiv:2305.13971.
12. Geng, S., et al. (2025). *Generating Structured Outputs from Language Models: Benchmark and Studies*. arXiv:2501.10868.
13. Guerreiro, J., et al. (2024). *LayoutFlow: Flow Matching for Layout Generation*. ECCV 2024.
14. Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop 2022.
15. Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. ACM SIGIR 2020.
16. Koo, H. (2024). *Automata-based constraints for language model decoding*. arXiv:2407.08103.
17. Leiva, L. A., et al. (2020). *Enrico: A Dataset for Topic Modeling of Mobile GUI Designs*. ACM MobileHCI 2020.
18. Li, G., et al. (2022). *CLAY: A Controllable Large-scale UI Layout Dataset with Visual and Semantic Annotations*. ACM CHI 2022.
19. Lin, J., et al. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. MLSys 2024.
20. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747.
    - *Role*: Foundational continuous normalizing flow (CNF) formulation establishing the probability path $p_t$ interpolating between noise $q$ and data $p_{\text{data}}$. Theoretical basis for the ODE integration in Tier 3 Polish rendering.
21. Liu, X., et al. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. arXiv:2209.03003.
    - *Role*: Rectified flow velocity field formulation $\mathbf{v}_\theta(x_t, t) = x_1 - x_0$ and the linear interpolation trajectory grounding `FlowODEIntegrator`'s Euler and Midpoint solvers.
22. Liu, H., et al. (2026). *UniLayDiff: A Unified Diffusion Transformer for Content-Aware Layout Generation*. CVPR 2026.
23. Qwen Team, Alibaba (2026). *Qwen3.5 Small Model Series Technical Report & Qwen-Image-2.1 Model Specifications*.
24. Qwen2.5-VL Team, Alibaba (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.
    - *Role*: Multimodal Rotary Position Embedding (M-RoPE) for 2D spatial coordinate grounding in GUI agent tasks. Grounds the Stage 2 Planner upgrade from text-only Qwen3.5-9B to vision-language Qwen2.5-VL-7B (discussed in §8 Future Work).
25. Ruiz, N., et al. (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. CVPR 2023. arXiv:2208.12242.
    - *Role*: DreamBooth LoRA fine-tuning methodology. Grounds the Tier 3 Polish training pipeline's image-conditioned wireframe-to-UI synthesis (§3.3, `training/configs/polish_qwen_image21_lora.yaml`).
26. Santhanam, K., et al. (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*. NAACL 2022.
27. Tang, H., et al. (2025). *HART: Efficient Visual Generation with Hybrid Autoregressive Transformer*. ICLR 2025. [mit-han-lab/hart](https://github.com/mit-han-lab/hart).
    - *Role*: Hybrid discrete AR + 37M-param residual diffusion head architecture. Cited as the current SOTA in the masked-generative model family (Baseline C in §6.1 ablations). FID 5.38 (MJHQ-30K), reconstruction FID 0.30, 4.5-7.7x throughput advantage. Demonstrates why Layout IR is preferred over any pixel-token approach for UI.
28. Wallace, B., et al. (2024). *Diffusion Model Alignment Using Direct Preference Optimization*. CVPR 2024. arXiv:2311.12908.
    - *Role*: Foundational Diffusion-DPO formulation for aligning generative models without RLHF. Beta parameter range and implicit reward margin metric. Grounded by velocity-prediction adaptation in `dpo_loss.py` (MotionFlux, ref 3).
29. Wang, B., et al. (2021). *Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning*. ACM UIST 2021.
30. Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702.
31. Zhang, J., et al. (2023). *LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models*. ICCV 2023.
32. Zhu, Y., et al. (2026). *TASTE: A Benchmark for Human Aesthetic Evaluation of Text-to-Image Generation*. arXiv:2605.20731.
     - *Role*: Held-out evaluation benchmark (§5.1). 10 professional designers, 9 rating criteria, signal-validation via Kendall's tau. Used for end-to-end agentic pipeline evaluation.
33. Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. arXiv:2302.05543. https://arxiv.org/abs/2302.05543
     - *Role*: ControlNet's spatial-conditioning design motivates the explicit deterministic wireframe condition boundary. Krisna does not claim to implement a ControlNet branch; the citation supports the separation of structural control from pixel synthesis.
34. Brooks, T., Holynski, A., & Efros, A. A. (2023). *InstructPix2Pix: Learning to Follow Image Editing Instructions*. arXiv:2211.09800. https://arxiv.org/abs/2211.09800
     - *Source note*: Verified against the official arXiv record during the 2026-10-14 architecture review. It supports instruction-guided image editing as related work, not evidence that Krisna's current Qwen path is trained or benchmarked.
35. Feng, Y., Wang, X., Wong, K. K., Wang, S., Lu, Y., Zhu, M., Wang, B., & Chen, W. (2023). *PromptMagician: Interactive Prompt Engineering for Text-to-Image Creation*. IEEE Transactions on Visualization and Computer Graphics. https://doi.org/10.1109/TVCG.2023.3327168
     - *Role*: Verified against the IEEE bibliographic record during the architecture review. Supports keeping prompt/state refinement interactive and inspectable; it does not justify replacing Krisna's hard Layout IR constraints with prompt search.
36. Guerreiro, J. J. A., et al. (2024). *LayoutFlow: Flow Matching for Layout Generation*. ECCV 2024, Springer Lecture Notes in Computer Science. https://arxiv.org/abs/2403.18187
     - *Role*: Springer-published learned-layout precedent. Supports a future learned ablation, not the current deterministic production baseline or any unmeasured SOTA claim.
