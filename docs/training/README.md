# Krisna Training Documentation

Comprehensive reference and mathematical derivations for training the generative components of Krisna's agentic UI pipeline.

> [!IMPORTANT]
> **Canonical Specification**: Defined in [docs/PRD.md §6](file:///d:/Krisna/docs/PRD.md#6-model-stack-the-no-rlhf-loop-revision) (The No-RLHF Model Stack) and [docs/PRD.md §8](file:///d:/Krisna/docs/PRD.md#8-data-pipeline-data-forge).
> Of the five model tiers, exactly **two** are trained; three ship frozen.

---

## 1. Trained Generative Components

| Component | Architecture | Base Model / Checkpoint | Training Methodology | Target Checkpoint |
|---|---|---|---|---|
| **Sketch Tier** | Bidirectional MaskGIT Transformer | From scratch | Progressive 2-Stage Cross-Entropy with CFG Dropout (10%) | `checkpoints/sketch_stage2_512/checkpoint_final.pt` |
| **Polish (Default)** | Single-Stream DiT (S3-DiT) | `Tongyi-MAI/Z-Image-Turbo` | DreamBooth LoRA (1024px) + Diffusion-DPO Velocity Alignment (512px) | `models/dpo_checkpoints/<run>/final/` |

---

## 2. Mathematical Foundations

### 2.1 Sketch Tier: Progressive ViT Positional Interpolation
To scale from Stage 1 ($256 \times 256$, grid $16 \times 16 = 256$ tokens) to Stage 2 ($512 \times 512$, grid $32 \times 32 = 1024$ tokens), Krisna employs bicubic interpolation of learned 2D positional embeddings prior to prefix token concatenation:

$$\mathbf{P}_{512} = \text{BicubicInterpolate}\left(\mathbf{P}_{256}, \text{size}=(32, 32)\right)$$

This preserves spatial layout geometry learned during Stage 1 without training 512px from scratch, verified in `training/src/krisna_training/sketch/pos_embed.py`.

### 2.2 Classifier-Free Guidance (CFG) & Caption Regularization
During training, conditioning dropout is enforced at rate $p_{uncond} = 0.10$. At inference, sampling uses Muse's linear ramped CFG schedule (Chang et al. 2023):

$$\ell_g = (1 + t)\ell_c - t\ell_u, \quad t \in [0, \gamma]$$

where $\ell_c$ represents logits conditioned on active prompt/constraints, $\ell_u$ represents unconditional logits, and $\gamma$ is the target guidance scale.

Captions are mixed at a 95/5 ratio:
- **95%**: Dense VLM-generated descriptions capturing layout hierarchy and widget semantics.
- **5%**: Original short dataset prompts to ensure robustness against concise user inputs (Betker et al. 2023).

### 2.3 Polish Tier: Flow-Matching Velocity DPO
Z-Image-Turbo is a flow-matching model predicting velocity $v_\theta(x_t, t, c)$ along a straight optimal transport probability path between noise $x_1 \sim \mathcal{N}(0, I)$ and clean latent $x_0$:

$$x_t = (1 - t)x_0 + t x_1, \quad v^* = x_1 - x_0$$

Direct Preference Optimization (DPO) aligns velocity predictions on human preference pairs $(x^w, x^l)$ without RLHF loops (Wallace et al. 2024, Bin et al. 2025):

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x^w, x^l), t} \left[ \log \sigma \left( \beta \cdot \left( \Delta \mathcal{E}_{\text{policy}} - \Delta \mathcal{E}_{\text{ref}} \right) \right) \right] + \lambda_{\text{anchor}} \mathcal{L}_{\text{anchor}}$$

where the squared error differences are:

$$\Delta \mathcal{E}_{\text{policy}} = \|v_\theta(x_t^l, t, c) - v^{*, l}\|^2 - \|v_\theta(x_t^w, t, c) - v^{*, w}\|^2$$

$$\Delta \mathcal{E}_{\text{ref}} = \|v_{\text{ref}}(x_t^l, t, c) - v^{*, l}\|^2 - \|v_{\text{ref}}(x_t^w, t, c) - v^{*, w}\|^2$$

$$\mathcal{L}_{\text{anchor}} = \|v_\theta(x_t^w, t, c) - v_{\text{ref}}(x_t^w, t, c)\|^2$$

---

## 3. Hardware Execution & Memory Budgets

| Task | Target Device | Precision | Peak VRAM | Offload Strategy |
|---|---|---|---|---|
| Sketch Stage 1 (256px) | RTX A6000 (48GB) | BF16 / FP32 | ~12 GB | Pure VRAM resident |
| Sketch Stage 2 (512px) | RTX A6000 (48GB) | BF16 | ~22 GB | Pure VRAM resident |
| Z-Image-Turbo LoRA (1024px) | RTX A6000 (48GB) | BF16 | ~32 GB | Gradient checkpointing + AdamW 8-bit |
| Z-Image-Turbo DPO (512px) | RTX A6000 (48GB) | BF16 | ~28 GB | Live VAE encoding + frozen reference model |

---

## 4. Runbooks & Detailed Guides

- **[TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)**: End-to-end execution guide on Intel Core i9 + NVIDIA RTX A6000 under Windows PowerShell.
- **[training/README.md](file:///d:/Krisna/training/README.md)**: Package setup, dataset preparation, and command-line execution walkthrough.
- **[SYNC_DESIGN.md](file:///d:/Krisna/docs/architecture/SYNC_DESIGN.md)**: Data-Forge to Training sync contract.

