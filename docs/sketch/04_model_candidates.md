# Model candidates and research review

## LayoutFlow — primary baseline

LayoutFlow (ECCV 2024) uses a roughly 15M, four-layer Transformer flow model with 512 hidden size and 8 heads. It supports unconditional generation, type/size conditioning, completion, and refinement through masks. On the authors’ RICO split it reports FID 2.37 and completion FID 1.51, but these are layout-feature metrics and are not directly comparable to image FID. Official code and checkpoints exist:

- paper: https://arxiv.org/abs/2403.18187
- code: https://github.com/JulianGuerreiro/LayoutFlow
- weights/data: https://huggingface.co/JulianGuerreiro/LayoutFlow

Adaptation required: replace generic element categories with Krisna UI types, add hierarchy and lock masks, and train against the Krisna IR. The released checkpoints are not treated as Krisna production checkpoints; the official repository notes a Python/PyTorch stack and custom Layout-FID requirements that must be ported and remeasured.

## LayoutDiffusion/LayoutDM — discrete baselines

LayoutDiffusion (ICCV 2023) models heterogeneous layout attributes with a legality-aware discrete diffusion process and plug-in conditional generation. LayoutDM (CVPR 2023) is a useful discrete baseline. These models are valuable ablations, but LayoutFlow is the better first implementation because completion and refinement are native and inference is faster.

- https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV2023_paper.pdf
- https://arxiv.org/abs/2305.02567

## UniLayDiff — content-aware upgrade

UniLayDiff (CVPR 2026) uses an MM-DiT formulation for background/content/layout constraints and adds relation conditioning through LoRA. It is relevant if the Sketch must place assets against a background image, but its reported 59M size and graphic-poster benchmarks make it a second-stage candidate, not the initial 12GB UI solution.

- https://arxiv.org/abs/2512.08897

## LayoutCoT — Planner baseline

LayoutCoT (2025/2026) reports training-free layout-aware RAG plus iterative reasoning. This is the correct zero-training baseline for comparing Planner-generated layouts against a trained layout model. Its reported benchmarks include RICO, WebUI, PubLayNet, PKU, and CGL, but its use of external LLMs means latency/cost and local-model behavior must be measured separately.

- https://arxiv.org/abs/2504.10829

## TA-TiTok/MaskGen — visual token alternative

TA-TiTok provides compact 1-D discrete or continuous tokens; MaskGen uses 128 discrete or 32 continuous tokens and CLIP conditioning. The released generators are 568M/1.1B, so they are not a drop-in small-model replacement for the current ~44M Sketch model. The tokenizer is still a credible experiment for reducing token count and improving semantic reconstruction.

- https://arxiv.org/abs/2501.07730
- https://github.com/bytedance/1d-tokenizer
