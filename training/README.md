# Krivex Training Architecture & Specification

## Canonical Architectural Model (PRD §6: Single-Trainable-Tier)

Krivex establishes a decoupled, mathematically bounded agentic design architecture with **a single trainable foundation model**. The entire training pipeline is dedicated exclusively to adapting the **Visual Polish Tier** (`Qwen/Qwen-Image-2.1`), while the **Planner** and **Sketch** tiers operate strictly as frozen, deterministic engines.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               TIER TRAINABILITY MATRIX                                 │
├────────────────────┬─────────────────────────────┬──────────────────┬──────────────────┤
│ Tier               │ Engine / Base Model         │ Status           │ Rationale        │
├────────────────────┼─────────────────────────────┼──────────────────┼──────────────────┤
│ Tier 1: Planner    │ Qwen3.5-9B-Instruct GGUF    │ 100% FROZEN      │ Grounded via RAG │
│                    │ (Unsloth W4A16 / AWQ)       │ (Zero training)  │ & CFG grammar    │
├────────────────────┼─────────────────────────────┼──────────────────┼──────────────────┤
│ Tier 2: Sketch     │ krivex.layout.v1            │ 100% DETERMIN-   │ 0 MB GPU VRAM;   │
│                    │ (Simplex / Cassowary CPU)   │ ISTIC & FROZEN   │ 0% overlap proof │
├────────────────────┼─────────────────────────────┼──────────────────┼──────────────────┤
│ Tier 3: Polish     │ Qwen-Image-2.1 (7B MMDiT)   │ ACTIVE TRAINABLE │ Adapts pixels to │
│                    │ (Pinned Diffusers efe46a5)  │ (QDoRA / DPO)    │ layout wireframe │
└────────────────────┴─────────────────────────────┴──────────────────┴──────────────────┘
```

---

## 1. Why the Planner and Sketch Tiers are Frozen

### 1.1 Tier 1: Planner (Frozen Vision-Language-Action)
- **Zero AI-Judge RLHF Loops**: In accordance with PRD §6 ("The No-RLHF-Loop Architectural Revision"), using one model's subjective aesthetic judgments to train another introduces severe reward hacking, catastrophic aesthetic decay, and conversational drift.
- **Empirical RAG Grounding**: Domain knowledge is supplied dynamically via the **UICrit Multi-Vector Visual RAG Index** (`planner_rag.py`, ColPali/CLIP multi-vector retrieval), pulling human-verified design guidelines directly into prompt context.
- **Pushdown Automata GBNF Grammar**: Schema validity is enforced mathematically at generation time using Context-Free Grammar (CFG) logit masking (Geng et al., EMNLP 2023). Invalid schema tokens are masked with $-\infty$ prior to softmax, guaranteeing $100\%$ valid `krivex.delta.v1` JSON deltas without parameter fine-tuning.
- **Gated DeltaNet Stability**: Fine-tuning Qwen3.5 with low-rank quantization impairs its hybrid linear recurrence memory (Gated DeltaNet). The frozen W4A16 GGUF preserves maximum long-context reasoning.

### 1.2 Tier 2: Sketch (Deterministic Neuro-Symbolic Engine)
- **0 MB GPU VRAM Guarantee**: Bounding box geometry, container hierarchy, and z-ordering are solved on the CPU via the dual-simplex algorithm (Cassowary, Badros et al., ACM TOCHI 2001), consuming $0\text{ MB}$ of VRAM and $<5\text{ms}$ latency.
- **Elimination of Codebook Collapse**: Discrete token codebooks (MaskGIT / VQGAN) permanently discard sub-pixel typography anti-aliasing and 1px border strokes under $16\times$ downsampling. The deterministic `krivex.layout.v1` rasterizer produces mathematically exact 2D TrueType wireframes and NVIDIA SPADE segmentation maps without neural hallucination.
- **Legacy Purge**: All historical neural layout ablations (`LayoutFlow` neural ODE and discrete MaskGIT training code) have been decommissioned from the codebase.

---

## 2. Visual Polish Tier (`Qwen-Image-2.1`) Training Architecture

In strict accordance with **PRD §6 ("Single-Trainable-Tier")** and **FINAL_ARCHITECTURE_VERDICT.md (§1, §2)**, Krivex does **NOT** require a complex 4-phase sequential training regimen. There is **EXACTLY ONE canonical production training target**: the **Supervised Polish LoRA/DoRA adapter (`polish-default`)**.

The other modules in `training/src/krivex_training/polish/` represent **offline scientific ablations** and **optional/deferred post-convergence research branches**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      CANONICAL PRODUCTION TRAINING TARGET                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Primary Target: Supervised QDoRA Base Adaptation (`polish-default`)                   │
│   (Wireframe PNG + Target Screenshot + Dual VLM/Human Caption)                         │
│   ├── Architecture: Qwen-Image-2.1 (7B Multimodal Diffusion Transformer)               │
│   ├── Supervision: Flow-matching velocity MSE on paired wireframe -> screenshot data  │
│   ├── Target Sublayers: 7 linear projections across attention and MLP                  │
│   ├── Hardware Profiles: A6000 (48GB BF16) & Consumer (12GB NF4)                      │
│   └── Artifact: checkpoints/qwen_image21_lora/                                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                OPTIONAL / DEFERRED POST-CONVERGENCE RESEARCH BRANCHES                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Benchmark Suite: Ablation Evaluation Matrix (`ablation_runner.py`)                     │
│   ├── Status: Scientific benchmark evaluation (NOT a sequential training phase)        │
│   └── Compares: Standard LoRA vs DoRA, MLP layers, +/- wireframe, +/- RAG              │
│                                                                                        │
│ Extension A: Flow-Matching Direct Preference Optimization (`polish-dpo`)              │
│   ├── Status: DEFERRED (Gate 5 in FINAL_ARCHITECTURE_VERDICT.md; PRD §6 optional)      │
│   ├── Data: 2,200 Human & Verifier Preference Pairs from preference_pairs.db           │
│   └── Formulation: Velocity error margin optimization (beta=2000, anchor=0.1)          │
│                                                                                        │
│ Extension B: Consistency Flow Matching (CFM) Distillation (`polish-cfm`)               │
│   ├── Status: EXPERIMENTAL FAST PREVIEW (Few-step student distillation)                │
│   ├── Goal: Distill 30 ODE steps down to 4-6 steps (~0.9s on A6000) for preview UI     │
│   └── Formulation: Pseudo-Huber consistency trajectory error (Song et al., ICLR 2024)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations

### 3.1 Continuous Flow Matching with Rectified Vector Fields
Let $x_0 \sim \mathcal{N}(0, I)$ be Gaussian noise and $x_1 \sim q(x_1)$ be the target screenshot latent. The linear probability trajectory is:
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
The constant target velocity field is:
$$u_t(x_t \mid x_0, x_1) = x_1 - x_0$$
The network $v_\theta(x_t, t, c)$ is trained via the mean squared error flow-matching loss:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim p(t), x_0, x_1, c} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]$$
using logit-normal timestep sampling $t = \sigma(\mu + \sigma \epsilon)$ ($\mu=0, \sigma=1$) to concentrate updates in the structural formation regime $t \in [0.2, 0.8]$.

### 3.2 Weight-Decomposed Low-Rank Adaptation (QDoRA)
To avoid quantization distortion when adapting frozen 4-bit base weights $W_0^{\text{NF4}}$, DoRA decouples the directional matrix from its magnitude scalar:
$$W = m \frac{W_0^{\text{NF4}} + \frac{\alpha}{r} B A}{\|W_0^{\text{NF4}} + \frac{\alpha}{r} B A\|_F}$$
where $r=64$, $\alpha=32$, and updates target all 7 MMDiT sublayer projections:
$$\mathcal{W}_{\text{target}} = \{\text{to\_q}, \text{to\_k}, \text{to\_v}, \text{to\_out.0}, \text{img\_mlp.gate\_layer}, \text{img\_mlp.proj}, \text{img\_mlp.out}\}$$

### 3.3 Flow-Matching Direct Preference Optimization (DPO)
Reward-free preference alignment optimizes velocity fields directly over winner-loser pairs $(x_0^w, x_0^l)$:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( -\beta \left( \|v_\theta(x_t^w, t) - v^*\|^2 - \|v_{\text{ref}}(x_t^w, t) - v^*\|^2 - (\|v_\theta(x_t^l, t) - v^*\|^2 - \|v_{\text{ref}}(x_t^l, t) - v^*\|^2) \right) \right) \right]$$
anchored by an optimal transport velocity loss to maintain generative fidelity.

---

## 4. Multi-Subject Composition & Localized Face Shielding (SOTA Literature)

The training pipeline integrates spatial routing principles from peer-reviewed foundations:
- **MS-Diffusion (IEEE/CVF CVPR 2024)**: Decoupled cross-attention spatial gating $\mathbf{A}_k = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}_k^T}{\sqrt{d}}\right) \odot \mathbf{M}_k$, preventing feature leakage and identity blending across distinct subjects in duo/group compositions.
- **FastComposer (ACM SIGGRAPH Asia 2023 / ACM TOG)**: Localized spatial masks isolating reference embeddings.
- **AnyDoor (IEEE/CVF CVPR 2024)** & **ConsisID (IEEE/CVF CVPR 2025 Highlight)**: Localized Crop $\to$ Lanczos Resample $\to$ Paste decoupling source crop coordinates $\mathbf{b}_k^s$ from target canvas placement $\mathbf{b}_k^t$, eliminating off-center facial drift ($0.00\%$ drift, $1.000$ lock score).
- **PhotoMaker (IEEE/CVF CVPR 2024)**: Strict $\ge 128\times 128\text{px}$ spatial bounding box floor preventing face token collapse in $16\times$ downsampling VAE / MMDiT patches.

---

## 5. Configuration Directory Structure

```
training/
├── README.md                                 # This architectural guide
├── pyproject.toml                            # Build & dependency metadata
├── configs/
│   ├── polish_qwen_image21_lora.yaml         # Primary Production: Full-precision A6000 (48GB) profile
│   ├── polish_qwen_image21_dpo.yaml          # Deferred Extension: Flow-Matching DPO profile
│   ├── cfm_distill_qwen_image21.yaml         # Fast Preview Extension: Consistency distillation profile
│   └── ablations/
│       ├── polish_qwen_qdora_12gb.yaml       # Primary Production: Consumer 12GB NF4 profile
│       ├── polish_qwen_adalora_48gb.yaml     # Adaptive rank allocation profile
│       ├── ablation_a_lora_attn.yaml         # Ablation A: Attention-only LoRA
│       ├── ablation_b_dora_attn.yaml         # Ablation B: Attention-only DoRA
│       ├── ablation_c_dora_no_wireframe.yaml # Ablation C: Text-only baseline
│       └── ablation_d_dora_no_rag.yaml       # Ablation D: No RAG conditioning
└── src/krivex_training/
    ├── __init__.py                           # Package specification & trainability matrix
    ├── polish/                               # Active Polish Tier training modules
    │   ├── prepare_dataset.py                # Pair dataset preparation & manifest builder
    │   ├── qwen_trainer_contract.py          # Pinned Diffusers trainer contract
    │   ├── train_dpo.py                      # Flow-Matching DPO trainer (deferred extension)
    │   ├── dpo_loss.py                       # DPO loss formulations & margin scoring
    │   ├── cfm_loss.py                       # Consistency Flow Matching distillation loss
    │   └── ablation_runner.py                # Automated ablation benchmark runner
    ├── data_forge_bridge/                    # Data-Forge export sync adapters
    │   ├── sync_polish_default.py            # Polish pair synchronization
    │   ├── sync_dpo_pairs.py                 # Preference pair SQLite synchronization
    │   ├── sync_planner_rag.py               # UICrit RAG corpus synchronization
    │   └── sync_sketch_tier.py               # Sealed deterministic stub
    ├── dpo/ & preference/                    # Preference pair SQLite storage & schema
    ├── planner/                              # 100% Frozen Planner module
    │   └── __init__.py                       # Sealed frozen specification
    └── sketch/                               # 100% Deterministic Sketch module
        └── __init__.py                       # Sealed deterministic specification
```

---

## 6. Execution Runbook

### 6.1 Pre-Flight Inspection (Read-Only)
```bash
python scripts/training/audit_training_pipeline.py
```
Reports active model IDs, source contracts, realized dataset pair counts, and hardware readiness without modifying disk state or GPU memory.

### 6.2 Synchronize Exported Data from Data-Forge
```bash
# Sync Polish pairs, preference pairs, and Planner RAG corpus:
python scripts/data-forge/sync_to_training.py --data-root D:\data_krivex
```

### 6.3 Primary Production Target: Launch Supervised Polish DoRA Training

The primary training configuration (`training/configs/polish_qwen_image21_lora.yaml`) trains Weight-Decomposed Low-Rank Adaptation (DoRA, rank 64, alpha 32) across the calibrated 7-bucket aspect ratio manifold (`1024,1024;768,1344;1344,768;832,1248;1248,832;896,1152;1152,896`), reducing geometric distortion on mobile (9:16) and web (16:9) to $\le 1.5\%$.

Supports dual operational profiles:
- **Fast Smoke/Test Profile (Default)**: 5,000 steps ($80,000$ exposures $= 0.365$ epochs on 218,791 pairs; $\sim 2.7\text{h}$ on RTX A6000 Ada).
- **Production SFT Convergence Profile**: 27,500 steps ($440,000$ exposures $= 2.01$ epochs; $\sim 15.2\text{h}$ on RTX A6000 Ada; optimal straight-line velocity adaptation window per Esser et al., ICML 2024).

```bash
# 1. Fast Smoke/Test Run (5,000 steps default):
./scripts/training/train_polish_default_lora.sh training/configs/polish_qwen_image21_lora.yaml

# 2. Production SFT Convergence Run (27,500 steps):
./scripts/training/train_polish_default_lora.sh training/configs/polish_qwen_image21_lora.yaml \
  --max_train_steps 27500 --lr_warmup_steps 500

# Windows (Native PowerShell Dispatcher)
powershell -ExecutionPolicy Bypass -File .\scripts\training\train_all.ps1 -Tier polish-default
```


### 6.4 Deferred Research Extension: Launch Flow-Matching DPO Preference Training
> [!CAUTION]
> **Advisory on DPO Preference Data Domain:**
> The active DPO training pipeline requires true condition-image-paired triplets $(x_w, x_l, c)$. Unconditioned general photography datasets (`pickapic_v2`, `hpdv2`) lack wireframe condition images and risk catastrophic forgetting of wireframe conditioning layers if run naively on top of the SFT adapter. Run Phase 2 DPO **only** after Phase 1 convergence, utilizing synthetic UI preference pairs generated via Resident Critic spatial perturbation.

```bash
# Linux / WSL
./scripts/training/train_polish_dpo.sh training/configs/polish_qwen_image21_dpo.yaml

# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\training\train_all.ps1 -Tier polish-dpo
```

### 6.5 Fast Preview Research Extension: Launch Consistency Flow Matching Distillation
```bash
# Linux / WSL
./scripts/training/train_polish_fast_preview.sh training/configs/cfm_distill_qwen_image21.yaml

# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\training\train_all.ps1 -Tier polish-cfm
```

### 6.6 Run Empirical Ablation Benchmark Matrix
```bash
python -m krivex_training.polish.ablation_runner --all
```

---

## 7. Hardware Requirements & Ledgers

| Profile | Target GPU | VRAM Budget | Base Precision | Adapter Precision | Method |
|---|---|---|---|---|---|
| **Enterprise / Workstation** | NVIDIA RTX A6000 (48GB) | 38.4 GB | BF16 (unquantized) | BF16 | DoRA ($r=64, \alpha=32$) |
| **Consumer Desktop** | NVIDIA RTX 4070 / 4080 (12–16GB) | 11.2 GB | 4-bit NF4 | BF16 | QDoRA ($r=64, \alpha=32$) |
| **Preview Distillation** | NVIDIA RTX A6000 / 4080 | 14.0 GB | BF16 / NF4 | FP16 / BF16 | CFM Distillation ($N=4\to 6$) |

All trained checkpoints generate cryptographic SHA-256 manifests (`manifest.json`) verifying adapter weights, base revision compatibility, and dataset provenance before inference serving.
