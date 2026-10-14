# Krisna Final Training-Pipeline Audit and Architecture Verdict

**Audit date:** 2026-10-13
**Scope:** repository implementation, training launchers/configuration, data contracts, model-selection proposal, planner/polish synchronization, and research-paper validity.
**Canonical decision:** proceed with the decoupled Planner → Layout IR → Polish architecture, but **do not proceed with the previously proposed five-week training plan or publish the current numerical results as measured results**.

This document is the final decision record. It supersedes unqualified model-stack and training claims in older architecture notes. It separates four kinds of evidence:

1. **Implemented:** directly supported by code in this checkout.
2. **Primary-source supported:** supported by an official model card or research paper.
3. **Design recommendation:** technically motivated, but not yet an empirical result.
4. **Pending validation:** requires a real data build, model smoke test, or GPU experiment.

## 1. Executive verdict

### Decision

**Proceed, with one trainable adapter, a frozen Qwen3.5-9B Planner, and a deterministic structure baseline.** The final research system should be:

```text
Qwen3.5-9B (frozen Planner; image-grounded review contract)
        │ image + text + Layout IR / verifier facts
        ▼
krisna.delta.v1  ──► deterministic Layout IR / wireframe
        │
        ▼
Qwen-Image-2.1 (Polish Default: base frozen, LoRA trained only after gate)
        │
        ▼
exact lock compositor + OCR/layout verifiers

Qwen-Image-2.1 without an adapter (Polish Quality ablation)
```

Data-Forge additionally uses a separate local `Qwen/Qwen3.8-27B` FP8 VLM for
preprocessing and auditing. It is an annotation/runtime dependency, not a
third trained model and not a source of training labels.

The Planner and Polish model are both Qwen-family models, but they **must not exchange hidden states**. Their synchronization boundary is an auditable contract: validated JSON deltas, exact text literals, normalized geometry, rasterized wireframe pixels, lock masks, and verifier receipts. This is more reproducible and safer than attempting an undocumented cross-model embedding bridge.

### What is approved now

- Keep the single-responsibility decomposition.
- Keep Layout IR and deterministic wireframe rendering as the authoritative geometry path.
- Freeze the Planner for the main paper experiment; do not spend research budget on Planner fine-tuning.
- Train only a Polish LoRA, and only after a real paired dataset and a one-batch trainer smoke test succeed.
- Use human/public preference data only for a later alignment experiment; do not use the Critic to manufacture labels.
- Use Qwen3.5-9B as the final Planner candidate because the official card documents a native vision-language architecture, image-text-to-text inference, structured multimodal serving, and a 9B/10B-parameter envelope compatible with the project runtime. Keep the exact GGUF/mmproj image request as a mandatory smoke gate.
- Retain Qwen2.5-VL-7B-Instruct only as a controlled historical/baseline comparison; do not silently mix its benchmark claims with Qwen3.5 results.

### What is rejected or deferred

- **Rejected:** treating the current `125K` value as an existing paired wireframe→screenshot corpus. The checkout contains the pair-preparation helper, not the realized pairs; the current workspace has no exported model-data pairs.
- **Rejected:** claiming that the Qwen trainer has completed a real run. The external Diffusers script and CLI contract are now pinned/audited, but the checkout still lacks the external source, target-GPU smoke test, checkpoint, and training receipt.
- **Deferred:** Diffusion-DPO/KTO until supervised Polish training, preference provenance, and the model-specific flow objective are independently validated.
- **Deferred:** HART, LayoutFlow, FLUX, and a learned Sketch generator. They are ablations or future work, not justified replacements for the current executable baseline.
- **Qualified:** exact protected-pixel claims now apply to the binary post-compositor path and its tests; seam quality remains a separate metric, and a real Qwen generation run is still pending.
- **Rejected:** visual-review claims for the current Planner path. The review prompt mentions a wireframe, but the audited request is text-only.

## 2. Audited model and trainability matrix

| Component | Current code-selected artifact | Current implementation status | Final research decision | Trainable? |
|---|---|---|---|---|
| Preprocessing VLM | `Qwen/Qwen3.8-27B` FP8, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Strict local Data-Forge runtime; not a training artifact | Use for annotation/quality/structure/audit only; never treat its outputs as human labels | **No** |
| Planner | `Qwen3.5-9B`, GGUF source aligned to official revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Frozen at inference; image-grounded review contract is implemented, real GGUF/mmproj smoke test pending | Primary Planner for this paper; Qwen2.5-VL-7B is a controlled baseline only | **No** |
| Structure/Sketch | Deterministic Layout IR backend; old docs also describe MaskGIT/Sketch training | Current learned Sketch source/entry point is incomplete or absent; Layout IR inference is real. | Make deterministic Layout IR the paper baseline. Add learned LayoutFlow only as a separately implemented ablation. | **No for main system** |
| Polish Default | `Qwen/Qwen-Image-2.1` revision `d26bb61231c349cf6b7896fa83353113880e1ba3` plus optional LoRA | Strict native QwenImage21Pipeline serving is wired; the actual Qwen trainer is still absent/unverified | Train one Qwen-Image-2.1 adapter only after trainer/API/data gates pass | **LoRA only; base frozen** |
| Polish Quality | Also `Qwen/Qwen-Image-2.1` | Factory passes an adapter only for Polish Default; the no-adapter reference still needs an explicit regression test. | Frozen no-adapter reference/ablation. Keep adapter injection tier-specific. | **No** |
| Legacy DPO | Historical Qwen/Z-Image path | Launcher exits early; `dpo_loss.py` source is absent in the checkout according to the audit. | Do not include in the primary pipeline. Restore as a separate experiment only after clean implementation. | **Deferred** |
| Critic/reviewer | Resident Planner review in current design; older docs mention Gemma | No independent Critic is required for the main no-RLHF experiment. | Use deterministic verifiers and optionally frozen VLM judging for evaluation only; never as training-label authority. | **No** |

### Important licensing decision

The official Hugging Face card for `Qwen/Qwen-Image-2.1` currently labels it **`qwen-research`** and describes a Qwen Research License Agreement. That is acceptable for a research-paper experiment only after the team records the exact license text and confirms the intended use. It must not be described as Apache-licensed or assumed product-safe. `Qwen/Qwen-Image-Edit-2511` is a separate 20B image-editing model with an Apache 2.0 card; it is a valid quality-path comparison, not the same model and not a drop-in proof that Qwen-Image-2.1 training works.

## 3. What the repository actually implements

### 3.1 Polish training: partial, not end-to-end validated

The local code implements strict filename-based pair preparation in `training/src/krisna_training/polish/prepare_dataset.py`. It writes `image`, `condition_image`, and `caption` metadata and rejects missing target/condition matches. That is useful and should be retained.

The launcher `scripts/training/train_polish_default_lora.sh` then constructs:

```text
third_party/diffusers/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py
```

The script is not present in this checkout. The setup helper now targets a
pinned Diffusers commit and the launcher has been audited against its official
CLI, but the external script is still not vendored or executed here. Therefore:

- pair preparation is implemented;
- the Qwen trainer objective is not implemented locally;
- the external script's existence is not verified locally and its runtime behavior is not GPU-validated;
- the model-specific conditioning path is not proven to consume `condition_image`;
- the trainable/frozen parameter counts are not recorded;
- a completed adapter checkpoint is not present in `models/`.

The YAML changes to rank 64, warmup, cosine scheduling, and 5,000 steps are reasonable **starting hypotheses**, not validated hyperparameters. Five thousand steps is not automatically sufficient for a 125K corpus; training must be scaled by effective batch and target passes after the realized count is known.

### 3.2 Data: structure sources are not automatically editing pairs

RICO and CLAY are valuable for structure, bounds, hierarchy, and rendering. They are not automatically `condition_image → target_image` editing pairs. A screenshot plus annotations becomes a Polish pair only after a deterministic wireframe renderer creates the condition image and the original screenshot is designated as the target. That join must be materialized, hashed, split, and counted.

The required pair record is:

```json
{
  "pair_id": "...",
  "source_id": "...",
  "condition_image": "...",
  "target_image": "...",
  "caption": "...",
  "split": "train|validation|test",
  "condition_sha256": "...",
  "target_sha256": "...",
  "license_source": "..."
}
```

The source identity split must happen before captioning, rendering, preference-pair generation, or VLM annotation. Split by application/page/source identity, not by individual image file.

### 3.3 Planner review contract is now multimodal, but not GPU-validated

The Planner path now accepts the active `wireframe_ref`, validates the image,
and attaches a base64 image content block to the llama.cpp request. Strict real
mode rejects a missing image or missing multimodal projector. An `mmproj` file
being discoverable remains insufficient evidence by itself; the real GGUF/
mmproj request still needs a target-GPU smoke test.

The final implementation must choose one of these contracts:

1. **Multimodal Planner:** attach the wireframe as an image content block, use the selected Qwen-VL processor/backend, and test that the request contains the image; or
2. **Text-grounded Planner:** retain as a controlled ablation with serialized Layout IR, OCR, geometry, lock, and verifier facts as text.

The paper's primary architecture will use option 1, while option 2 remains a controlled ablation.

### 3.4 Locking is currently approximate

`polish_backend_qwen.py` creates a binary mask, applies `GaussianBlur(radius=8)`, and composites with the source. Boundary pixels are blended. The current implementation therefore cannot support the paper's claim of `Δ_lock = 0` over the whole locked region.

Final semantics:

- `protected_mask`: copied source pixels, exact equality;
- `seam_ring`: optional blend region outside the protected mask;
- evaluation mask: protected pixels only, with explicit half-open pixel-coordinate convention.

## 4. Final Planner–Polish synchronization contract

The Qwen models are coordinated through artifacts, not shared internal representations.

### Planner output

The Planner receives:

- current user message and bounded history;
- current Layout IR and/or rendered wireframe;
- immutable constraints and lock records;
- verifier facts from the previous render;
- optional UICrit retrieval results, marked as retrieval context rather than labels.

It returns a grammar-constrained `krisna.delta.v1` object containing:

- exact user-intent changes;
- additions/removals/updates to Layout IR nodes;
- literal text strings;
- normalized boxes and parent IDs;
- lock/unlock operations;
- a short user-facing reply;
- retrieval IDs for provenance.

### Deterministic merge and render

The orchestrator validates schema, IDs, bounds, tree acyclicity, and lock policy before merging. It then renders the authoritative wireframe from Layout IR. The renderer—not either model—owns geometry, text literals, hierarchy, and lock coordinates.

### Polish input

Polish receives:

- the deterministic wireframe image;
- the effective natural-language style prompt;
- a machine-readable lock mask;
- the same Layout IR hash and revision ID;
- a deterministic seed and sampling configuration.

Polish may add texture, lighting, material, and decorative detail. It may not redefine locked geometry, exact text literals, node hierarchy, or normalized coordinates. A post-compositor and verifier enforce these rules.

### Review loop

The review output must cite the Layout IR revision and verifier receipt it inspected. A review delta that has no corresponding node ID or metric evidence is rejected. The system must log whether review was image-grounded or text-grounded; the two modes must not be pooled silently in results.

## 5. Final training plan

### Gate 0 — make the experiment reproducible

Before a full run, freeze:

- model IDs, exact revisions, licenses, and SHA256 manifests;
- Diffusers/Transformers/PEFT commits and Python/CUDA versions;
- one authoritative launcher for Linux and Windows;
- dataset source revisions and post-filter counts;
- app/page-level split manifests;
- seeds, image resolution, sampler, and output naming;
- trainable/frozen parameter report;
- checkpoint selection rule;
- a `training_receipt.json` for every run.

Do not clone `latest` for a paper experiment.

### Gate 1 — Planner smoke test

Compare Qwen3.5-9B and Qwen2.5-VL-7B-Instruct on a fixed 100-screen internal corpus:

- JSON schema validity;
- node/type extraction F1;
- box IoU/gIoU;
- text literal extraction CER;
- locked-node preservation;
- contradiction handling;
- latency, VRAM, and image-token throughput;
- image-attached versus text-only review.

Select the primary Planner by a predeclared score and memory threshold, not by model recency. No Planner fine-tuning is needed for the main paper.

### Gate 2 — build the real paired corpus

1. Ingest eligible RICO/CLAY records and structural annotations.
2. Split by application/page/source identity.
3. Render each valid annotation to a deterministic wireframe.
4. Pair wireframe condition with the original screenshot target.
5. Reject malformed trees, missing screenshots, duplicate IDs, unreadable files, and license-ineligible records.
6. Deduplicate using source identity plus image hash/perceptual hash.
7. Write the pair manifest, counts, hashes, and licenses.
8. Keep synthetic/VLM captions in a separate provenance field; never report them as human labels.

The number of usable pairs is an output of this gate. Do not call it 125K until the manifest proves it.

### Gate 3 — validate the Qwen-Image-2.1 adapter path

The trainer must be either pinned and vendored or invoked from a pinned external commit. The smoke test must prove all of the following:

- Qwen-Image-2.1 loads through its native documented pipeline class/API;
- the condition image reaches the model as an image-conditioning input;
- the target image is the supervision target;
- base Qwen parameters have `requires_grad=False`;
- only declared LoRA parameters have gradients;
- one optimizer step changes LoRA weights;
- save/load restores the adapter;
- inference uses the same pipeline family and conditioning contract;
- manifest validation rejects the wrong base model or wrong adapter type.

The serving path uses the native `QwenImage21Pipeline`; the pinned training
contract is audited against the official image-to-image script, but the exact
installed Diffusers API and one-batch runtime behavior still require target-GPU
validation.

### Gate 4 — supervised Polish LoRA

Start with a 1–5% stratified corpus smoke run, then scale to the full manifest. Suggested initial sweep:

| Variable | Initial values |
|---|---|
| LoRA rank | 16, 32, 64 |
| Learning rate | 5e-5, 1e-4 |
| Effective batch | 4, 8, 16 |
| Target passes | 1, 2, 3 |
| Edit strength | fixed for training; sweep only at eval |
| Seed | at least 3 for the selected configuration |

Primary early-stopping endpoint: exact-text CER plus locked-region equality and layout adherence. Aesthetic score alone is not sufficient for UI generation.

### Gate 5 — preference alignment, only after the baseline

Use real human preference pairs from eligible public datasets, with source/license manifests and no AI-judge labels in the training signal. For a flow-matching model, do not silently reuse DDPM epsilon-prediction DPO. Implement and unit-test the velocity objective, then run a beta sweep such as `{250, 500, 1000, 2000}` with a frozen reference model and an anchor term ablation.

This stage is optional for the main paper. If the Qwen-Image-2.1-specific preference trainer cannot be made reproducible, report supervised LoRA as the final system and preference alignment as future work.

### Gate 6 — no learned Sketch claim in the main paper

The deterministic Layout IR baseline is a strength of the system, not a deficiency. It isolates geometry from pixel generation and makes lock/text metrics interpretable. Train LayoutFlow or HART only as a separately implemented ablation after the main pipeline is stable; do not compare a hypothetical HART result with the current deterministic renderer as if both were measured systems.

## 6. Evaluation plan for a defensible paper

### Primary endpoints

1. **Structure:** node/type F1, matched box IoU and gIoU, parent/tree accuracy, relation accuracy, legality rate.
2. **Instruction following:** exact requested edit success, constraint satisfaction, contradiction handling, delta validity.
3. **Typography:** CER/WER with fixed OCR engine, preprocessing, language, scale, and seed.
4. **Locking:** exact pixel equality inside `protected_mask`; boundary/seam quality reported separately.
5. **Perceptual quality:** KID/FID only with a declared feature space and matching reference distribution; CLIP/vision-language scores are secondary.
6. **Human preference:** blind randomized pairwise evaluation with rater expertise, sample size, criteria, agreement, and confidence intervals.
7. **Runtime:** cold/warm latency, peak allocated/reserved VRAM, host RAM, throughput, and failure/fallback rate.

### Public VLM/UI reporting

Qwen2.5-VL's published report provides a useful model capability reference, including OCR, spatial localization, and agent benchmarks such as ScreenSpot. Those are **Planner capability baselines**, not end-to-end UI-polish scores. Report them separately from Krisna's task-specific evaluation.

Do not use:

- Layout-FID as image FID;
- CLIP directional alignment as proof of exact instruction following;
- a VLM judge as the only quality metric;
- a benchmark score without split, prompt, seed, evaluator version, and sample count;
- a model's public benchmark score as a prediction of Krisna's end-to-end score.

### Required ablations

- Qwen2.5-VL text-only vs image-grounded Planner review;
- Qwen3.5-9B vs Qwen2.5-VL-7B, if both pass the smoke test;
- frozen Qwen-Image-2.1 vs supervised LoRA;
- condition wireframe vs no condition image;
- Layout IR vs screenshot-only prompt;
- exact lock compositor vs feathered compositor;
- RAG on/off;
- NF4/BF16 Planner inference, with quality and memory reported separately;
- preference alignment off/on, only if that trainer is executable.

## 7. Negative-point audit

| Proposed claim/design | Verdict | Reason |
|---|---|---|
| “Qwen-Image-2.1 can be trained with the current launcher” | **Not validated** | The external trainer contract is pinned/audited, but the source, target-GPU run, checkpoint, and receipt are absent. |
| “RICO+CLAY already provide ~125K edit pairs” | **False as a repository state claim** | They provide structure/screens; pairing requires a separate renderer/join and manifest. |
| “All Qwen-family components synchronize naturally” | **Misleading** | Preprocessing VLM, Planner, and Polish are different runtime/model families/tasks; synchronization must be through explicit artifacts. |
| “Planner visually reviews the wireframe today” | **Implemented contract; runtime pending** | The call now attaches a validated image payload, but target-GPU GGUF/mmproj execution is not yet measured. |
| “Locked region MSE is guaranteed zero” | **False for current code** | Gaussian feathering changes edge pixels. |
| “DPO is ready for the Qwen polish model” | **Not validated** | Current DPO launcher/source path is incomplete; flow objective is model-specific. |
| “HART is the necessary Sketch replacement” | **Unsupported for this project** | HART is a pixel-generation architecture; Krisna's main geometry path is deterministic Layout IR. |
| “FLUX.2 is automatically better” | **Unsupported** | Any replacement needs matched UI data, resolution, conditioning, license, and task-specific evaluation. |
| “Current paper table proves superiority” | **Rejected** | No run manifest, artifacts, split, seeds, confidence intervals, or independent reproduction is attached. |

## 8. Research-paper disposition

The existing `docs/RESEARCH_PAPER.md` is retained as a historical formal specification, but its reported result tables and hardware numbers must be treated as **provisional placeholders** until receipts exist. The paper cannot currently claim measured IoU 0.865, OCR CER 0.047, zero lock drift, 10.2 GB peak, zero swap latency, statistical parity, or superiority over baselines.

The paper should publish the final system in this order:

1. system definition and exact checkpoint manifest;
2. data provenance, licenses, split IDs, and pair count;
3. model train/freeze table;
4. Planner–Layout IR–Polish synchronization contract;
5. training objective and adapter receipt;
6. task-specific evaluation protocol;
7. measured results with uncertainty;
8. negative results and limitations;
9. ablations and public benchmark context.

Until those artifacts exist, the correct paper status is **research specification and pre-registration**, not validated empirical paper.

## 9. Primary sources used for the verdict

These are the sources that support the decision, not claims that a future experiment has already succeeded.

1. **Qwen Team, `Qwen/Qwen2.5-VL-7B-Instruct` model card.** [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). Supports the controlled multimodal Planner baseline and image-message input.
2. **Qwen Team, `Qwen/Qwen2.5-VL-7B-Instruct` model card.** [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). Supports the concrete checkpoint, documented Transformers usage, image message format, and published model-card evaluation context.
3. **Qwen Team, `Qwen/Qwen3.5-9B` model card.** [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B). Supports the native vision-language architecture, image-text-to-text contract, 9B parameter scale, 262K context, and vLLM/SGLang serving guidance. It does not by itself prove GGUF/mmproj compatibility or Krisna task quality.
4. **Qwen Team, `Qwen/Qwen-Image-2.1` model card.** [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1). Supports the 7B visual-generation component, 32 single-stream DiT layers, native `QwenImage21Pipeline`, image editing, image conditioning, supported resolutions, CPU offload example, and Qwen Research License Agreement. It does not provide a validated DreamBooth-LoRA trainer for this repository.
5. **Qwen Team, `Qwen/Qwen-Image-2.1` official model card and blog.** [Model card](https://huggingface.co/Qwen/Qwen-Image-2.1), [official blog](https://qwen.ai/blog?id=qwen-image-2.1). Supports the native pipeline, image-conditioning contract, supported resolutions, and license.
6. **Qwen Team, `Qwen/Qwen-Image-Edit-2511` model card.** [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). Supports the separate 20B image-editing comparison and Apache 2.0 license; it must not be conflated with Qwen-Image-2.1.
7. **Esser et al., “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis,” ICML 2024.** [PMLR 235](https://proceedings.mlr.press/v235/esser24a.html). Supports rectified-flow training, resolution-aware noise-scale design, and multimodal transformer conditioning.
8. **Brooks et al., “InstructPix2Pix: Learning To Follow Image Editing Instructions,” CVPR 2023.** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html). Supports instruction-conditioned image editing as a related-work precedent.
9. **Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV 2023.** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html). Supports a future explicit spatial-control ablation while keeping the base generator frozen.
10. **Kulikov et al., “FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models,” ICCV 2025.** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html). Supports a future inversion-free flow-editing comparison, not a Qwen result.
11. **Feng et al., “PromptMagician: Interactive Prompt Engineering for Text-to-Image Creation,” IEEE TVCG 2023.** [IEEE DOI](https://doi.org/10.1109/TVCG.2023.3327168). Supports inspectable prompt/state artifacts, not a replacement for Layout IR constraints.

## 10. Completion criteria

The architecture is ready for the research run only when all are true:

- one launcher and one manifest define the active stack;
- the Planner sends and receives a tested image-grounded contract;
- the pair manifest contains nonzero, license-eligible, source-disjoint pairs;
- the Qwen-Image-2.1 native trainer passes one-batch train/save/load/inference tests;
- the Quality path cannot load the Default adapter accidentally;
- real inference cannot silently report synthetic fallback as Qwen output;
- exact lock tests pass over every protected pixel;
- every paper number points to a receipt with checkpoint, split, seed, evaluator, and hardware;
- the final paper labels estimates, design targets, and measurements separately.
