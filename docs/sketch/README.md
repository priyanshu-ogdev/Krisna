# Sketch/layout next-stage design

## Current model

The repository currently trains a project-owned MaskGIT-lineage image-token model:

- `SketchModelConfig`: 16,384-token VQGAN vocabulary
- 12 bidirectional Transformer encoder layers
- hidden size 512, 8 heads, FFN size 2,048
- learned absolute positional embeddings
- CLIP ViT-L/14 768-dim prefix conditioning
- token logits plus Token-Critic confidence head
- Stage 1: 256px, 16x16 grid, from scratch
- Stage 2: 512px, 32x32 grid, positional interpolation from Stage 1

It uses `boris/vqgan_f16_16384`, masked-token cross entropy, Token-Critic BCE, cosine masking during training, and Halton/confidence reveal during inference.

## Recommendation

Do not replace it blindly. Make **structured layout generation the primary Sketch contract**, retain the current MaskGIT model as a visual baseline, and evaluate a compact 1-D token baseline only if a visual draft remains product-critical.

Primary candidate: a LayoutFlow-style continuous flow model with a UI-specific typed layout representation and hard constraint projection.

Secondary candidate: UniLayDiff-style multimodal diffusion/flow architecture if background/image/content conditioning is required.

Visual fallback candidate: TA-TiTok/MaskGen-style compact 1-D tokens. The published MaskGen models are 568M/1.1B, not small replacements for the current 44M model, so use their tokenizer ideas first rather than assuming the released generator fits the hardware target.

## Stage folders

- `01_current_model.md` — implementation facts and known limitations
- `02_layout_ir.md` — proposed UI layout intermediate representation
- `03_data_contract.md` — real-data-only training and annotation policy
- `04_model_candidates.md` — LayoutFlow, LayoutDiffusion, UniLayDiff, LayoutCoT, MaskGen
- `05_training_plan.md` — training objectives and curriculum
- `06_inference_contract.md` — Planner → layout → renderer → Polish handoff
- `07_evaluation.md` — metrics and ablations
- `08_rollout.md` — migration and rollback gates
