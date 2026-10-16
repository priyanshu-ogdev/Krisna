# Model candidates and research review

## LayoutFlow — primary baseline

LayoutFlow (ECCV 2024) uses a roughly 15M, four-layer Transformer flow model with 512 hidden size and 8 heads. It supports unconditional generation, type/size conditioning, completion, and refinement through masks. On the authors’ RICO split it reports FID 2.37 and completion FID 1.51, but these are layout-feature metrics and are not directly comparable to image FID. Official code and checkpoints exist:

- paper: Springer LNCS ECCV 2024, doi: 10.1007/978-3-031-72764-1_4
- code: https://github.com/JulianGuerreiro/LayoutFlow
- weights/data: https://huggingface.co/JulianGuerreiro/LayoutFlow

Adaptation required: replace generic element categories with Krisna UI types, add hierarchy and lock masks, and train against the Krisna IR. The released checkpoints are not treated as Krisna production checkpoints; the official repository notes a Python/PyTorch stack and custom Layout-FID requirements that must be ported and remeasured.

## LayoutDiffusion/LayoutDM — discrete baselines

LayoutDiffusion (ICCV 2023) models heterogeneous layout attributes with a legality-aware discrete diffusion process and plug-in conditional generation. LayoutDM (CVPR 2023) is a useful discrete baseline. These models are valuable ablations, but LayoutFlow is the better first implementation because completion and refinement are native and inference is faster.

- IEEE/CVF ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV2023_paper.pdf
- IEEE/CVF CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/papers/Chai_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.pdf

## InstructLayout & UniLayDiff — structured prior upgrades

InstructLayout (IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3595880) formulates instruction-driven layout generation conditioned on structured semantic graph priors, justifying Krisna's structured delta decomposition and graph prefix injection in `planner_backend_llama.py`. UniLayDiff (IEEE/CVF CVPR) uses an MM-DiT formulation for background/content/layout constraints and adds relation conditioning through LoRA.

- IEEE TPAMI 2025: https://doi.org/10.1109/TPAMI.2025.3595880

## Visual Token Alternatives & Multimodal Grounding

SeeClick (ACL 2024, doi: 10.18653/v1/2024.acl-long.505) and CogAgent (IEEE/CVF CVPR 2024, pp. 14351–14361) establish screen-space visual coordinate grounding and dual-stream visual attention for GUI agents.

- ACL 2024: https://aclanthology.org/2024.acl-long.505.pdf
- IEEE/CVF CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_CogAgent_A_Visual_Language_Model_for_GUI_Agents_CVPR_2024_paper.pdf

