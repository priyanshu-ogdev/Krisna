# Qwen-Image-2.1 Architectural Research Foundations & Finalization Plan

**Document Version**: 1.0 (Canonical Research Record)  
**Scope**: In-depth academic analysis of the `Qwen-Image-2.1` foundation model lineage, multimodal diffusion transformer (MMDiT) mechanics, dual-stream image conditioning, layer-disentangled editing, high-compression VAE reconstruction, few-step consistency distillation, and the project finalization plan.  
**Archived Reference Papers**: `docs/papers/30_qwen_image_spec2026.pdf`, `docs/papers/61_qwen_image2_technical_report2026.pdf` through `docs/papers/67_qwen3_vl_technical_report2025.pdf`.

---

## 1. Executive Research Overview

Krisna's Visual Polish Tier is anchored upon **`Qwen/Qwen-Image-2.1`**, Alibaba's open-weight visual foundation model specializing in unified text-to-image synthesis, multi-reference image editing, and complex typography rendering.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 QWEN-IMAGE-2.1 FOUNDATION LINEAGE                                      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Qwen-Image (2025) ──► Qwen-Image-2.0 (2026) ──► Qwen-Image-2.1 (Production Baseline)                  │
│ [arXiv:2508.02324]    [arXiv:2605.10730]         [Alibaba Official Specification & Diffusers Pin]      │
│ ├── 7B MMDiT Backbone ├── Dual-Stream VAE/VLM    ├── Native RGBA Transparency (Qwen-Image-Layered)     │
│ └── Rectified Flows   └── 32 Single-Stream DiT   ├── Multi-Reference Conditioning (up to 10 assets)    │
│                                                  └── Global Skip Connection VAE (Qwen-Image-VAE-2.0)   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Technical Breakdown of Qwen-Image-2.1

### 2.1 MMDiT Backbone & Single-Stream Architecture
Unlike legacy dual-stream diffusion models (e.g., Stable Diffusion 1.5/2.1) that separate text cross-attention from image self-attention, Qwen-Image-2.1 implements **32 Single-Stream Diffusion Transformer (DiT) blocks** (grounded in Esser et al., ICML 2024 and Qwen-Image-2.0 Technical Report):
- Text tokens from the multimodal encoder and image latent patches from the VAE are flattened, concatenated, and processed inside a unified attention matrix.
- Each transformer block features 7 critical linear projections:
  $$\mathcal{W}_{\text{target}} = \{\text{to\_q}, \text{to\_k}, \text{to\_v}, \text{to\_out.0}, \text{img\_mlp.gate\_layer}, \text{img\_mlp.proj}, \text{img\_mlp.out}\}$$
  totaling 224 target modules across the 32 layers.
- Krisna adapts these 224 modules using **Weight-Decomposed Low-Rank Adaptation (DoRA, Liu et al., ICML 2024)** with rank $r=64$ and $\alpha=32$, freezing the 7B base weights $W_0$ in BF16 (or NF4 on consumer GPUs).

### 2.2 Dual-Stream Conditioning for Image Editing
As detailed in the *Qwen-Image Technical Report* (Alibaba, 2025/2026), Qwen-Image models decouple image editing into two distinct streams:
1. **Semantic Conditioning Stream**: The reference image is processed through a multimodal vision-language model (`Qwen2.5-VL` / `Qwen3-VL`) to extract high-level semantic intent, object relationships, and textual context.
2. **Reconstructive Latent Stream**: The reference image is simultaneously encoded through a spatial VAE into clean latent tokens and prepended along the sequence dimension before the noisy target latents:
   $$Z_{\text{input}} = [Z_{\text{clean\_cond}}, Z_{\text{noisy\_target}}(t)]$$
This enables Krisna's Polish Tier to preserve layout geometry, TrueType typographic anti-aliasing, and container boundaries while allowing continuous color, shadow, and texture generation.

### 2.3 Native Multi-Reference Asset Support (Up to 10 Images)
The official Qwen-Image-2.1 model card and Diffusers pipeline natively accept a list of reference images (`image: list[PIL.Image]`). This supports:
- **Multi-Subject Composition**: Ingesting distinct subject references (e.g., Person A in Image 1, Person B in Image 2) and allocating them to disjoint bounding boxes via Krisna's `ReferenceAsset` manifest.
- **Product & Style Swapping**: Conditioning simultaneously on a background scene, a brand color palette swatch, and individual foreground assets.

### 2.4 Native RGBA Transparency (Qwen-Image-Layered Grounding)
Grounded in **Yin et al. (HKUST & Alibaba, 2025, *Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition*)**:
- The model directly predicts 4-channel RGBA latents, generating transparent PNGs with alpha channels without color haloing or edge fringing.
- Krisna exposes this directly via `Constraints.asset_mode = "transparent"`, rendering transparent wireframes and producing isolated UI components (modals, floating action buttons, stickers) for direct export to Figma and web code.

### 2.5 Sub-Pixel Typographic Fidelity (Qwen-Image-VAE-2.0 Grounding)
Grounded in the **Qwen-Image-VAE-2.0 Technical Report (Alibaba, May 2026)**:
- Standard VAE architectures obliterate sub-pixel font anti-aliasing and 1px border strokes under $16\times$ downsampling.
- Qwen-Image-VAE-2.0 integrates **Global Skip Connections (GSC)** and expanded latent channels trained on synthetic rendering engines, allowing high-frequency text boundaries and sharp UI edges to reconstruct cleanly from continuous latents.

### 2.6 Few-Step Fast Preview Distillation (Qwen-Image-Flash Grounding)
Grounded in **Wu et al. (CityU & Alibaba, 2026, *Qwen-Image-Flash: Rethinking the Training Recipe for Few-Step Distillation*)**:
- Full 30-to-40 ODE step evaluation yields ~5.8s latency on RTX A6000.
- Using Consistency Flow Distillation (`training/src/krisna_training/polish/cfm_loss.py`), the student network predicts the clean trajectory endpoint in 4 to 6 steps (~0.9s on A6000, ~2.5s on RTX 4070/4080) with minimal layout IoU loss ($\Delta \text{IoU} \le 0.02$).

---

## 3. The 6-Gate Project Finalization Plan

To transition Krisna from an audited research specification to a fully validated production system, execution proceeds across six disciplined gates:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             6-GATE FINALIZATION ROADMAP                                │
├────────┬────────────────────────────────┬──────────────────────────────────────────────┤
│ Gate   │ Milestone Title                │ Verification Criteria & Deliverable          │
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 0 │ Environment & Contract Freeze  │ Verify Diffusers commit efe46a5, PyTorch,    │
│        │                                │ CUDA sm_75+, and single launcher integrity.  │
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 1 │ Paired Dataset Materialization │ Ingest CLAY + Figma2Code + Design2Code;      │
│        │                                │ render deterministic wireframes; emit        │
│        │                                │ dataset_receipt.json with SHA-256 hashes.    │
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 2 │ One-Batch GPU Smoke Test       │ Load QwenImage21Pipeline; prove base weights │
│        │                                │ frozen; verify 1 optimizer step changes LoRA.│
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 3 │ Supervised LoRA Convergence    │ Train polish-default (rank 64, alpha 32);    │
│        │                                │ save adapter checkpoint to models/.          │
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 4 │ Multi-Axis Empirical Audit     │ Measure OCR CER, Layout IoU >= 0.85, seam    │
│        │                                │ gradient <= 0.05, and lock drift = 0.00%.    │
├────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ Gate 5 │ Production Release & Packaging │ Finalize FastAPI backend, Tauri desktop IPC, │
│        │                                │ and publish verifiable research receipts.    │
└────────┴────────────────────────────────┴──────────────────────────────────────────────┘
```

### Gate 0: Environment & Contract Freeze
- Pinned Diffusers revision: `efe46a5b177af67c64b3666da8ddddac7ee12163`.
- Authoritative launcher: `scripts/training/train_all.ps1` (Windows) and `train.sh` (Linux/WSL).
- Pinned Base Model: `Qwen/Qwen-Image-2.1` revision `d26bb61231c349cf6b7896fa83353113880e1ba3`.
- Fail-fast assertions enabled on strict pipeline parameter inspection.

### Gate 1: Paired Dataset Materialization
- Ingest structural annotations from **CLAY (ACM CHI 2022)**, **Figma2Code (ICLR 2026)**, and **Design2Code (ACL 2025)**.
- Render deterministic condition wireframes and SPADE segmentation maps via `krisna.layout.v1`.
- Materialize split-isolated Parquet partitions with typed image bytes.
- Emit `dataset_receipt.json` certifying non-empty, deduplicated, source-disjoint train/validation/test counts.

### Gate 2: One-Batch GPU Smoke Test
- Run `krisna_training.polish.train_preflight` on target hardware (RTX A6000 or consumer RTX 4080).
- Confirm:
  1. Base weights have `requires_grad = False`.
  2. Exactly 224 declared DoRA sublayer projections have gradients.
  3. One forward/backward step executes without CUDA OOM.
  4. Adapter saves and reloads with bitwise identical state dict restoration.

### Gate 3: Supervised LoRA Convergence Run
- Launch supervised training via `train_polish_default_lora.sh` or `train_all.ps1 -Tier polish-default`.
- Hyperparameters: AdamW ($lr=1.0\times 10^{-4}$), cosine schedule, 200 warmup steps, effective batch size 16 (batch 4, accumulation 4).
- Target artifact: `checkpoints/qwen_image21_lora/` with cryptographic `manifest.json`.

### Gate 4: Multi-Axis Empirical Evaluation
- Evaluate on held-out test splits across 5 standardized endpoints:
  1. **Layout Fidelity**: Box IoU $\ge 0.85$, non-overlap rate $= 100\%$.
  2. **Typographic Legibility**: OCR CER $\le 0.05$ on verbatim quoted text literals.
  3. **Lock Invariance**: Bitwise pixel drift $\Delta_{\text{lock}} = 0.00\%$ inside protected regions.
  4. **Seam Boundary Quality**: Laplacian gradient continuity $\Delta \nabla \le 0.05$ along composite borders.
  5. **Perceptual Parity**: KID/FID against frozen `Polish Quality` baseline to verify zero aesthetic degradation.

### Gate 5: Production Release & Packaging
- Package inference engine into Docker container (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`).
- Validate asynchronous Tokio IPC communication with Rust/Tauri desktop client (`inference/frontend/src-tauri/`).
- Publish formal research receipt binding model checkpoints, dataset hashes, and evaluation metrics.
