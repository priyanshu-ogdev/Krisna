# Krisna SOTA Audit — October 2026

**Audit date:** 2026-10-15
**Scope:** frozen Planner, deterministic Layout IR, Qwen-Image-2.1 Polish
adapter, Data-Forge preprocessing, region locking, and research-paper claims.

This is a research scan, not evidence that any external method has been
reproduced in Krisna. It deliberately uses **publisher proceedings, official
model cards, and official implementation documentation only**. No arXiv-only
paper is used as evidence for an upgrade. A method is marked **candidate**
until its code, checkpoint, license, hardware footprint, and task-specific
evaluation are available.

## Executive decision

Do **not** replace the current mainline before the first governed experiment.
The highest-value upgrades are not another foundation-model swap; they are:

1. build the real source-disjoint wireframe-to-screenshot corpus;
2. run the native Qwen-Image-2.1 one-batch trainer gate;
3. run an official multimodal Qwen3.5 Planner smoke test;
4. make exact protected-region and typography metrics the primary endpoints.

The current deterministic Layout IR is a valid experimental control because it
separates geometry from pixel rendering. Replacing it with a learned layout
model before a baseline exists would remove the paper's most interpretable
ablation.

## Current stack versus relevant SOTA

| Area | Current Krisna choice | Relevant 2025/2026 evidence | Decision |
|---|---|---|---|
| Planner | Frozen Qwen3.5-9B with image-grounded review contract | The official Qwen3.5-9B card is an image-text-to-text model and documents Transformers, vLLM, and SGLang image requests. OSWorld (NeurIPS 2024) establishes the multimodal desktop evaluation standard. | Keep frozen. Validate official multimodal serving before treating the llama.cpp GGUF path as equivalent. |
| Layout | Deterministic `krisna.layout.v1` | InstructLayout (IEEE TPAMI 2025) proves semantic graph priors stabilize instruction-driven layout; LayoutFlow (ECCV 2024) demonstrates continuous flow matching. | Keep deterministic geometry. Any learned layout branch must be a separately implemented ablation with a publisher-backed source and matched evaluation. |
| Polish | Qwen-Image-2.1, frozen base plus one supervised LoRA | The official card documents native image editing, up to ten reference images, masks/annotations, 2K-class aspect ratios, and `QwenImage21Pipeline`. | Keep as the first trainable path. The native trainer is externally pinned and contract-audited, but not yet GPU smoke-tested. |
| Region locking | Binary exact compositor after generation | ControlNet (ICCV 2023 CVF) establishes spatial control through a trainable conditioning branch; FlowEdit (ICCV 2025 CVF) demonstrates inversion-free editing for pretrained flow models. | Preserve exact post-compositing as the hard guarantee. Treat native mask/control conditioning as a future controlled ablation, not a silent change. |
| Typography & OCR | Deterministic Layout IR literal mapping | TextDiffuser-2 (ECCV 2024) and AnyText (ICLR 2024) prove diffusion models require language-guided layout bounding boxes for legible text synthesis. | Enforce literal string mapping in Layout IR before pixel diffusion. Bounding boxes are mandatory condition inputs. |
| Sampling speed | Qwen native scheduler | Esser et al. (ICML 2024, PMLR 235) supports rectified-flow training/sampling and resolution-aware noise-scale design; the Qwen model card defines the native scheduler/API. | Do not transplant a sampler from another backbone. Benchmark native step counts and scheduler behavior on Qwen first. |
| Low-memory alternative | Qwen-Image-2.1 | The official Qwen card documents `enable_model_cpu_offload()` and supported 2K-class aspect ratios. | Improve the existing offload path and measure it before considering a different foundation model. |
| External comparison | None in the mainline | FLUX.2-dev is a 32B rectified-flow image generation/editing model with multi-reference support, but the HF card requires acceptance of a non-commercial license. | Use only as a controlled quality/compliance comparison; remove unsupported “current open-weight leader” and percentage-gain claims. |
| Planner memory | Bounded history plus UICrit RAG | Official Qwen3.5 documentation supports image-text-to-text serving; no publisher-backed memory-routing upgrade was used. | Measure visual-wireframe versus serialized-IR memory directly on the fixed Planner bake-off. |
| Structure/OCR preprocessing | Qwen3.8-27B VLM plus deterministic schema gates | The pinned Qwen3.8 model card and local vLLM contract are the current primary implementation evidence. | Keep strict Qwen3.8 preprocessing; do not mix an unverified parser into training data. |
| GUI grounding | Planner review plus deterministic geometry | ShowUI (CVPR 2025) demonstrates UI-guided visual token selection; SeeClick (ACL 2024) evaluates screen grounding coordinates. | Keep the current Planner frozen and measure grounding/review on the project corpus. |

## Significant upgrades recommended

### P0 — reproducibility and data, not a model swap

- Pin every Data-Forge model and encoder revision in the production profile.
- Materialize the Figma2Code/local source receipt, deterministic wireframe
  renders, pair manifest, hashes, licenses, and source-disjoint splits.
- Run the strict Qwen3.8-27B VLM pipeline on the target CUDA host and record
  peak VRAM, throughput, exclusions, and audit pass rate.
- Keep synthetic VLM outputs separate from human/source annotations and never
  call registry capacity a realized training count.

### P1 — validate the two current model contracts

- Planner: compare the existing llama.cpp GGUF/mmproj request with the official
  Qwen3.5 Transformers/vLLM or SGLang image request on the same 100-screen
  corpus. Measure JSON validity, box IoU, OCR CER, review latency, and memory.
- Polish: run one batch through the pinned Diffusers Qwen-Image-2.1 img2img
  trainer and prove that the condition image, target image, frozen base, LoRA
  gradients, save/load, and serving path agree.

### P1 — structured layout deliberation

Add a controlled LayoutCoT-style ablation without storing unrestricted hidden
chain-of-thought:

```text
retrieved layout exemplars + current Layout IR + verifier facts
    -> bounded structured review steps
    -> krisna.delta.v1
    -> deterministic merge and render
```

The comparison must be against the current Planner→Layout IR path, not against
an unimplemented learned layout generator.

### P2 — lock-aware polishing

Evaluate three conditions separately:

1. Qwen-Image-2.1 image conditioning plus exact post-compositor;
2. Qwen-Image-2.1 plus an explicit mask/control input if the native API supports
   it at the pinned revision;
3. a MaskFlow-inspired mask-aware training/control ablation.

Report protected-pixel equality and seam quality separately. Do not average a
seam score into an exact-lock metric.

### P2 — speed and deployment bake-off

Test ReAL-style adaptive stepping, native Qwen step counts, and any Qwen
distilled variant on the same seeds and pair set. Measure quality loss, OCR,
layout adherence, lock equality, peak VRAM, and wall-clock latency. A sampler
paper result on FLUX is not a Qwen result.

### P3 — alternative foundation models

Only after the Qwen baseline has a real receipt should Krisna compare Qwen-
Image-2.1 against FLUX.2-dev or Mage-Flow. The comparison must hold condition
images, prompts, resolution, seeds, evaluation masks, and training budget
constant, and must include license/deployment constraints.

## Claims that must not appear as facts

- “Data-Forge already has 125K Qwen pairs.” It does not.
- “RICO/CLAY counts equal paired editing examples.” They require rendering and
  joining a condition image to a target.
- “FLUX.2 is automatically better” or “25–35% better than Z-Image.” No matched
  Krisna benchmark or primary-source evidence supports that claim.
- “HART is required.” HART/MaskGen are relevant masked-generation research,
  but the main Krisna geometry path is deterministic Layout IR.
- “Qwen3.5 GGUF/mmproj is equivalent to official multimodal Qwen3.5 serving.”
  This remains a target-GPU compatibility question.
- “MaskFlow/ReAL/Mage-Flow results transfer to Qwen-Image-2.1.” They require
  model-specific validation.

## Sources checked

- Qwen3.5-9B model card: <https://huggingface.co/Qwen/Qwen3.5-9B>
- Qwen-Image-2.1 model card: <https://huggingface.co/Qwen/Qwen-Image-2.1>
- Pinned Diffusers Qwen-Image-2.1 training README:
  <https://raw.githubusercontent.com/huggingface/diffusers/efe46a5b177af67c64b3666da8ddddac7ee12163/examples/dreambooth/README_qwenimage21.md>
- Esser et al., “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis,” ICML 2024, PMLR 235: <https://proceedings.mlr.press/v235/esser24a.html>
- Brooks et al., “InstructPix2Pix: Learning To Follow Image Editing Instructions,” CVPR 2023 Open Access: <https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html>
- Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV 2023 Open Access: <https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html>
- Kulikov et al., “FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models,” ICCV 2025 Open Access: <https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html>
- Lin et al., “ShowUI: One Vision-Language-Action Model for GUI Visual Agent,” CVPR 2025: <https://doi.org/10.1109/CVPR52734.2025.01816>
- InstructLayout, “Instruction-Driven 2D and 3D Layout Synthesis with Semantic Graph Prior,” IEEE TPAMI 2025: <https://doi.org/10.1109/TPAMI.2025.3595880>
- Chen et al., “TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering,” Springer ECCV 2024: <https://doi.org/10.1007/978-3-031-72761-0_1>
- Xie et al., “OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024.
- FLUX.2-dev model card: <https://huggingface.co/black-forest-labs/FLUX.2-dev>

## Publication status after this audit

The repository is a **research specification with an unbuilt data/training
artifact**, not a completed empirical paper. The main paper becomes
experiment-ready only after the P0/P1 gates produce receipts. External SOTA
methods should be reported as related work or predeclared ablations until
those matched experiments exist.
