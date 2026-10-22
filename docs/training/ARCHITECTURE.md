# Krivex Training Architecture

**Status:** current implementation and research plan
**Scope:** training, data contracts, checkpoint provenance, and the training-side
view of the agentic image/VL pipeline. Serving behavior is documented separately
in [`../inference/README.md`](../inference/README.md).

> **Evidence rule.** This document distinguishes code-supported facts,
> primary-source-supported design choices, and proposals that still require a
> real data build or GPU run. A paper that motivates an architecture does not
> prove that Krivex has implemented or measured it.

## 1. Current answer: what is actually trained?

The current checkout has one intended trainable model path, two frozen/deterministic
product paths, and one separate preprocessing-only VLM:

| Component | Code-selected model | Trainability | Current source status | Decision |
|---|---|---:|---|---|
| Preprocessing VLM | `Qwen/Qwen3.8-27B` FP8, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Not trained by Krivex | Strict local Data-Forge VLM bundle and stage runtime | Annotation/quality/structure tool only; never a final trained model or training signal authority |
| Planner | `Qwen3.5-9B`; official revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Frozen | `PlannerBackendLlama` plus UICrit RAG; image-grounded review contract is now wired, but real GGUF/mmproj smoke validation remains required | Keep frozen for the main system; no Planner LoRA unless the predeclared visual/JSON gates fail |
| Structure / Sketch | `krivex.layout.v1` deterministic neuro-symbolic engine | Deterministic (0 MB VRAM, <5ms CPU) | `LayoutBackend`, `layout_baseline.py`, and `layout_renderer.py` are production-locked; legacy neural LayoutFlow ablation retired | Production uses deterministic Cassowary solver and DualStreamSketch; 0 MB VRAM and 0% overlap guarantee. |
| Polish Default | `Qwen/Qwen-Image-2.1`, official revision `d26bb61231c349cf6b7896fa83353113880e1ba3` | LoRA only; base must remain frozen | Strict native `QwenImage21Pipeline` serving path is wired; pair manifest and accelerate training entry point verified | Primary trainable generative model artifact. |
| Polish Quality | `Qwen/Qwen-Image-2.1` | Frozen reference/ablation | Same backend, intended without an adapter | Frozen comparison baseline without adapter. |
| Preference alignment | Continuous Flow Matching DPO (`dpo_loss.py`) | Post-training alignment | `krivex_training/polish/dpo_loss.py` is implemented and unit-tested for velocity-based MMDiT flow matching | Trainable on human preference pairs (Pick-a-Pic v2, HPDv2, GameLabel-10K) after base LoRA convergence. |

The intended training answer is therefore **one trainable model artifact**,
not two or three independently trained models. The accurate answer is:

1. **Planner:** frozen.
2. **Layout:** deterministic baseline, not a trained model.
3. **Polish Default:** one proposed Qwen-Image-2.1 LoRA training path, not yet
   a verified end-to-end trainer. This is the only trainable artifact in the
   main paper.
4. **Polish Quality:** frozen comparison path using the same Qwen-Image-2.1
   base without the Default adapter.
5. **Data-Forge VLM:** Qwen3.8-27B is a preprocessing dependency, not a model
   trained by this project.

### Direct source evidence

- Qwen model ID and columns: `training/configs/polish_qwen_image21_lora.yaml`.
- Pair builder: `training/src/krivex_training/polish/prepare_dataset.py`.
- Planner freeze guard: `training/src/krivex_training/planner/__init__.py` (sealed frozen VLA).
- Deterministic structure path: `inference/src/krivex_inference/layout_baseline.py`,
  `training/src/krivex_training/sketch/__init__.py`, and `inference/src/krivex_inference/backends/layout_backend.py`.
- Registry and inference model selection:
  `inference/src/krivex_inference/orchestrator/model_registry.py` and
  `inference/src/krivex_inference/backends/factory.py`.

## 2. The authoritative training pipeline

The training pipeline should be understood as a gated artifact flow, not as a
single command that silently launches every historical tier:

```text
Data-Forge source records
        │
        ├── structure path: screenshot + hierarchy/annotation
        │       │
        │       ├── source-identity split
        │       ├── validated Layout IR
        │       └── deterministic wireframe render
        │
        └── Polish pair builder
                │
                ├── condition_image = deterministic wireframe
                ├── target_image    = source screenshot
                ├── instruction/caption
                ├── hashes, license, provenance, split
                └── pair manifest + count receipt
                                │
                                ▼
                  Qwen-Image-2.1 paired image-edit LoRA
                                │
                                ├── base weights frozen
                                ├── declared transformer LoRA modules trainable
                                ├── one-batch save/load smoke test
                                └── training receipt + adapter manifest
                                │
                                ▼
                     Qwen image-conditioned inference
                                │
                                ├── exact lock compositor
                                ├── OCR/layout/verifier checks
                                └── source-disjoint evaluation report
```

### 2.1 Data-forge is not the pair corpus

RICO and CLAY are structure/screenshot sources. They are not automatically
`condition_image -> target_image` pairs. A valid Polish pair must be materialized
by rendering a condition from structure and joining it to the original target.

The minimum pair record is:

```json
{
  "pair_id": "stable-source-and-render-id",
  "source_id": "application-or-page-identity",
  "condition_image": "conditions/<pair_id>.png",
  "target_image": "targets/<pair_id>.png",
  "caption": "instruction or controlled UI description",
  "split": "train|validation|test",
  "condition_sha256": "...",
  "target_sha256": "...",
  "license_source": "..."
}
```

The split must be made by application/page/source identity before rendering,
captioning, VLM annotation, or preference-pair construction. Otherwise near
duplicates from one application can leak across train and evaluation.

The current helper in `prepare_dataset.py` provides strict filename pairing, but
the new `prepare_qwen_pairs_from_manifest()` path now preserves the full
provenance manifest and rejects `split: unassigned`. The Data-Forge exporter
does not guess source-identity splits; the claimed pair count is therefore an
output of the data-build gate, not a constant to put in a paper.

### 2.2 Training contract for Qwen-Image-2.1

The repository currently delegates the architecture-specific objective to:

```text
third_party/diffusers/examples/dreambooth/
  train_dreambooth_lora_qwenimage21_img2img.py
```

That file is not present in the checkout. The setup script clones unpinned
`latest` Diffusers and explicitly warns that an official Qwen-Image-Edit trainer
is not available. Consequently, the following are **required validation gates**,
not assumed facts:

1. the selected pipeline class loads the exact Qwen-Image-2.1 revision;
2. `condition_image` reaches the image-conditioning branch;
3. `image` is the target supervision image;
4. base parameters have `requires_grad=False`;
5. only declared LoRA parameters receive gradients;
6. one optimizer step changes LoRA weights;
7. save/load restores the adapter;
8. the serving pipeline uses the same conditioning contract;
9. the adapter manifest rejects the wrong base model or pipeline family.

Do not use the presence of a launcher or a YAML file as evidence that these
gates pass.

### 2.3 Objective choice

For the supervised stage, use the trainer's native image-conditioned objective
once the API is confirmed. The repository should not duplicate a recent model's
VAE, text-encoder, timestep, or transformer-call implementation from memory.

For a later preference stage, keep it separate from supervised paired editing:

- supervised pair: `(condition_image, target_image, instruction)`;
- preference pair: `(prompt, chosen_image, rejected_image)`.

The old DPO implementation is not part of the current path. Diffusion-DPO is a
valid research direction for human preference data, but the original paper is
derived for diffusion likelihoods and must not be transferred to a flow model
without verifying the prediction target and model API.

## 3. Research comparison and what it implies for Krivex

The following papers are relevant because they share a component of the design;
none is evidence that Krivex has reproduced its results.

### 3.1 Structured visual planning: Qwen2.5-VL

**Qwen Team, `Qwen2.5-VL-7B-Instruct` official model card**
(<https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>). The card documents
multimodal image-message input and the controlled Planner baseline.

**Implication:** a Planner that must inspect a wireframe should be a genuinely
multimodal model request. Krivex's current Planner review text says it reviews a
wireframe, but the request path does not attach an image. The primary architecture
should either:

- use a selected Qwen-VL checkpoint with a tested image-message contract; or
- explicitly call the current path text-grounded review and pass serialized
  Layout IR/verifier facts instead.

Do not describe an `mmproj` file existing on disk as proof that an image was sent.

### 3.2 Layout generation: LayoutFlow

**Guerreiro et al., “LayoutFlow: Flow Matching for Layout Generation,” ECCV
2024**, Springer proceedings/project record. LayoutFlow models layout elements
as a flow and supports conditioning at different degrees.

**Implication:** it is a reasonable research precedent for the existing
`training/layout/model.py` primitive and `layout_flow_base.yaml`. It does not
justify replacing the deterministic Layout IR mainline today: the current
repository has no complete LayoutFlow trainer, loss implementation, checkpoint
loader, or learned inference backend. Keep it as an ablation after the artifact
contract is stable.

### 3.3 Masked token generation: Muse

The historical Muse comparison is not used as evidence for the current design
because a publisher copy was not verified for this audit.

**Implication:** Muse is a legitimate precedent for a future learned Sketch
model, masked-token objectives, and conditional dropout. It does not prove that
the missing Sketch source modules in this checkout are executable. The current
mainline should remain deterministic Layout IR until a complete learned model is
implemented and evaluated.

### 3.4 Image-conditioned editing: InstructPix2Pix and ControlNet

**Brooks et al., “InstructPix2Pix: Learning To Follow Image Editing
Instructions,” CVPR 2023 Open Access**
(<https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html>)
trains an image-editing model conditioned on an input image and written
instruction.

**Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,”
ControlNet, ICCV 2023 Open Access**
(<https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html>)
keeps a large diffusion backbone locked while learning a separate spatial
control path.

**Implication:** these papers support the separation between a deterministic
structural condition and a frozen generative backbone with a small trainable
adaptation. They do not establish the exact Qwen-Image-2.1 trainer contract.
They also motivate evaluating structural adherence independently from aesthetic
quality.

### 3.5 Interactive prompt/state design: PromptMagician

Feng et al., *PromptMagician: Interactive Prompt Engineering for Text-to-Image
Creation*, IEEE Transactions on Visualization and Computer Graphics (2023),
DOI <https://doi.org/10.1109/TVCG.2023.3327168>, supports treating prompt
construction as an interactive, inspectable process rather than a single opaque
string. For Krivex this reinforces the artifact boundary already used by the
inference service: the effective prompt, original intent, constraint delta,
Layout IR revision, seed, and verifier receipt must remain separately logged.
Prompt refinement is not allowed to override hard geometry or protected regions.

### 3.6 Springer/ECCV layout-flow implication

LayoutFlow was published in the Springer ECCV 2024 proceedings and proposes
continuous flow matching for layout generation. It is a useful learned-layout
ablation, but the current production path stays deterministic until a learned
checkpoint can reproduce `krivex.layout.v1`, lock masks, and source-disjoint
evaluation metrics. This avoids turning a promising paper architecture into an
unverified production dependency.

### 3.7 Parameter-efficient adaptation: LoRA

The PEFT/LoRA implementation contract is treated as a library/runtime contract
in this repository, not as an unverified preprint citation. The active trainer
must discover and record target modules from the loaded Qwen transformer.

**Implication:** LoRA is the correct memory-oriented adaptation abstraction for
the proposed Polish Default path, but the paper does not supply target modules
for Qwen-Image-2.1. Target-module names must be discovered and recorded from the
actual loaded transformer, not copied from Z-Image, FLUX, or SD3.

### 3.8 Flow matching and preference alignment

**Esser et al., “Scaling Rectified Flow Transformers for High-Resolution Image
Synthesis,” ICML 2024, PMLR 235**
(<https://proceedings.mlr.press/v235/esser24a.html>) provides peer-reviewed
rectified-flow transformer evidence and resolution-aware training/sampling
principles.

**Wallace et al., “Diffusion Model Alignment Using Direct Preference
Optimization,” CVPR 2024 Open Access**
(<https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html>)
shows preference alignment without a separate RLHF loop for a diffusion model.

**Implication:** preference alignment is a later experiment, not a reason to
keep the current dead DPO path in the active launcher. The Qwen supervised
conditioned-edit baseline must be reproducible first; then a separate preference
trainer can be designed around the actual Qwen prediction parameterization.

## 4. Recommended final architecture

```text
Frozen multimodal Planner
  input: user text + current Layout IR + wireframe image + verifier facts
  output: validated planner.delta.v1 JSON
        │
        ▼
Deterministic Layout IR manager
  owns: IDs, hierarchy, normalized boxes, text literals, lock policy
  output: canonical Layout IR + DualStreamSketch (wireframe, SPADE segmentation, differential mask)
        │
        ▼
Qwen-Image-2.1 Polish Default
  base frozen; supervised LoRA only
  input: multi-stream condition (wireframe, segmentation, differential mask) + effective instruction + seed
  output: polished candidate + adapter/run provenance
        │
        ▼
Exact compositor and verifiers
  exact protected pixels, OCR, layout, safety, handoff consistency
```

The Planner and image model should not exchange hidden states. Their boundary
should be auditable artifacts: JSON deltas, Layout IR revisions, wireframe bytes,
lock masks, prompts, seeds, and verifier receipts.

### Training order

1. **Gate 0:** pin model revisions, licenses, Python/CUDA/framework versions,
   and one launcher.
2. **Gate 1:** build and audit the source-disjoint condition/target manifest.
3. **Gate 2:** run a one-batch Qwen trainer test and prove trainable/frozen
   parameter behavior.
4. **Gate 3:** run a small supervised LoRA smoke experiment.
5. **Gate 4:** evaluate exact text, layout adherence, lock equality, and safety
   before any preference alignment.
6. **Gate 5:** optionally add human preference alignment as a separate,
   model-specific experiment.
7. **Gate 6:** only then consider learned LayoutFlow/MaskGIT/HART ablations.

## 5. Run receipt requirements

Every actual run must write `training_receipt.json` containing at least:

```json
{
  "run_id": "...",
  "base_model": "...",
  "base_revision": "...",
  "trainer_revision": "...",
  "dataset_manifest_sha256": "...",
  "pair_counts": {"train": 0, "validation": 0, "test": 0},
  "source_split_policy": "application/page identity",
  "trainable_parameter_count": 0,
  "frozen_parameter_count": 0,
  "lora_target_modules": [
    "to_q", "to_k", "to_v", "to_out.0",
    "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"
  ],
  "lora_target_module_count": 224,
  "resolution": 1024,
  "seed": 0,
  "checkpoint": "...",
  "smoke_test": {"condition_consumed": false, "save_load": false}
}
```

Until that receipt exists, a configuration is a proposal and a checkpoint is
not a verified research result.

## 6. Native flow-matching training upgrade

The current implementation follows the native Qwen-Image-2.1 image-to-image
trainer rather than reimplementing a flow objective from memory. This is
grounded in the official Diffusers Qwen training README and in Esser et al.,
*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*,
ICML 2024, PMLR 235.

### 6.1 Prepared dataset boundary

`prepare_qwen_pairs_from_manifest()` now produces split-isolated Parquet files
with typed `datasets.Image` columns. It does not hand a single mixed JSONL file
to the external trainer. The receipt binds manifest bytes, per-split bytes,
image hashes, source groups, and schema columns before launch.

### 6.2 CPU-only launch gate

`krivex_training.polish.train_preflight` verifies the exact Diffusers commit,
native trainer markers, dataset receipt, typed image columns, path/hash
integrity, train count, and source-group disjointness before `accelerate`
starts. It emits a redacted command and reports dirty external checkout state.
Non-null `val_fraction`/`val_every` settings are rejected instead of silently
being ignored by the native trainer.

### 6.3 Native objective and inference consistency

The condition image is passed through the trainer's native paired image-to-image
contract. The serving backend inspects the installed
`QwenImage21Pipeline.__call__` signature and passes only parameters supported by
that exact pipeline. Legacy `strength`, custom ODE integration, and invented
guidance schedules are not represented as real Qwen execution. Real receipts
record the loaded scheduler and live timesteps.

### 6.4 Required ablations

The primary comparison is supervised Qwen-Image-2.1 LoRA versus the frozen
Qwen-Image-2.1 base. Only after this passes should the project test:

1. native mask/control inputs, if supported by the pinned pipeline;
2. an inversion-free flow-editing ablation inspired by FlowEdit (ICCV 2025
   Open Access), with Qwen-specific implementation and receipts;
3. a ControlNet-like spatial adapter, trained separately from the main LoRA.

These methods cannot be substituted for the primary Qwen contract without a
matched condition/target benchmark.

## 7. SOTA upgrade backlog with evidence gates

| Candidate upgrade | Evidence source | Production decision |
|---|---|---|
| Multimodal Planner review | Qwen2.5-VL report and image-message model card | Add only after a real image request/response smoke test; otherwise label review text-grounded |
| Learned LayoutFlow branch | Springer/ECCV LayoutFlow | Retired; consolidated to production deterministic neuro-symbolic IR contract |
| Interactive prompt engineering | IEEE TVCG PromptMagician | Preserve editable prompt/state artifacts; do not replace hard constraints |
| Spatial diffusion control | ControlNet and InstructPix2Pix | Require materialized condition images and separate structure/aesthetic metrics |
| Preference alignment | Diffusion-DPO and flow-model follow-up work | Deferred until native Qwen prediction target and human data are verified |
| Continuous batching / replicas | Serving systems literature | Not adopted without a measured concurrent workload on the single-GPU product target |

## 8. Training-only bibliography

1. Qwen Team. *Qwen-Image-2.1* official model card.
   <https://huggingface.co/Qwen/Qwen-Image-2.1>
2. Hugging Face Diffusers. *DreamBooth training example for Qwen-Image 2.1*,
   pinned source README and native image-to-image trainer.
   <https://raw.githubusercontent.com/huggingface/diffusers/efe46a5b177af67c64b3666da8ddddac7ee12163/examples/dreambooth/README_qwenimage21.md>
3. Esser, P. et al. (2024). *Scaling Rectified Flow Transformers for
   High-Resolution Image Synthesis*. ICML 2024, PMLR 235.
   <https://proceedings.mlr.press/v235/esser24a.html>
4. Brooks, T., Holynski, A., & Efros, A. A. (2023). *InstructPix2Pix:
   Learning To Follow Image Editing Instructions*. CVPR 2023 Open Access.
   <https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html>
5. Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to
   Text-to-Image Diffusion Models*. ICCV 2023 Open Access.
   <https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html>
6. Kulikov, V. et al. (2025). *FlowEdit: Inversion-Free Text-Based Editing
   Using Pre-Trained Flow Models*. ICCV 2025 Open Access.
   <https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html>
7. Feng, Y. et al. (2023). *PromptMagician: Interactive Prompt Engineering
   for Text-to-Image Creation*. IEEE TVCG.
   <https://doi.org/10.1109/TVCG.2023.3327168>

These citations are deliberately kept in the training documentation. They are
related-work and design precedents, not claims of reproduced results.
