# Phase 5: Agentic workflow design and research record

## Scope and decision

The production workflow uses one model per generative responsibility:

1. **Planner**: frozen Qwen planner artifact through llama.cpp. It maintains
   conversational intent, emits validated state deltas, and operates in design
   and review modes.
2. **Layout**: deterministic Layout IR projection and wireframe renderer. It
   is the geometry authority and is intentionally not replaced by a language
   or image model.
3. **Polish**: `Qwen/Qwen-Image-2.1` through `QwenImage21Pipeline`. Both
   Polish operating modes use this same architecture. A LoRA is optional and
   selected at startup; it is never selected as a runtime fallback.
4. **Verifier stack**: specialized safety, OCR, semantic, layout, aesthetic,
   and handoff-consistency evaluators. These score and gate output; they do
   not generate replacement images.

No additional generative model is justified until a measured product gap
cannot be addressed by one of these responsibilities.

## End-to-end contract

```text
user message
  -> Planner (validated JSON delta)
  -> DesignState constraint merge
  -> Layout IR + wireframe
  -> user iteration
  -> pixel handoff
  -> Qwen-Image-2.1 image-conditioned refinement
  -> safety gate and quality verifiers
  -> provenance-bearing final output
  -> Planner review mode
```

The Polish contract requires a handoff image. Blank-slate generation is not a
valid recovery path because it breaks the causal connection between Layout IR
and the final render.

## Why no additional model is currently needed

### No second image generator

Z-Image-Turbo and Qwen-Image-Edit-2511 represent incompatible pipeline and
training assumptions. Keeping either as a production fallback would make
quality comparisons and preference data ambiguous. Qwen-Image-2.1 is the
single image architecture for this phase.

### No separate Critic model

The Planner already supports review mode and consumes verifier scores,
conversation history, and prior review context. A second critique LLM would
add memory pressure and a second source of design decisions without an
established measured benefit.

### No learned Sketch model yet

Layout IR provides deterministic geometry, inspectable state, and stable
handoff semantics. A learned sketch model should only be added after an
evaluation shows that deterministic layout projection is the limiting factor.

## Training compatibility

The retired Z-Image flow-matching DPO loss is not used. Qwen-Image-2.1
adapters must be produced with the official image-conditioned DreamBooth LoRA
workflow using condition and target image columns.

Every adapter should ship a manifest with at least:

```json
{
  "base_model": "Qwen/Qwen-Image-2.1",
  "pipeline": "QwenImage21Pipeline",
  "conditioning": "image",
  "training_kind": "dreambooth_lora_img2img"
}
```

Inference rejects a manifest that names another base model or pipeline. A
missing or incomplete manifest is rejected before model allocation. This makes
adapter compatibility a load-time contract rather than an operator assumption.

## Resource and failure policy

The orchestrator may recover resources through CPU offload, allocator cleanup,
or an operator-approved retry. It must not substitute another image model.
Failure should restore Planner/Layout residency and return an explicit,
retryable error with the Qwen model identity.

The orchestrator now retries the requested Polish tier only. It does not
substitute a different renderer after a budget or runtime failure. This keeps
the causal handoff and model identity stable for both users and experiments.

## Provenance and reproducibility

Finalize output records the model ID, pipeline class, adapter path, inference
steps, seed, prompt used, handoff image, safety result, and verifier scores.
This makes preference-pair construction and research evaluation auditable.

## Evaluation plan

The paper should report separate metrics for:

- Planner constraint accuracy and multi-turn retention;
- Layout IR validity and geometric consistency;
- handoff consistency from wireframe to polished image;
- OCR readability;
- semantic prompt alignment;
- layout adherence;
- aesthetic preference;
- safety pass rate;
- Qwen latency, peak VRAM, CPU-offload cost, and failure rate;
- adapter-vs-base comparison using the same prompts, handoffs, seeds, and
  verifier versions.

Do not combine these into one unsupported scalar. Report the distribution and
the evaluation protocol, including model revisions and hardware.

## Open empirical questions

1. Actual Qwen-Image-2.1 peak VRAM at the selected resolution and 40 steps.
2. Whether CPU offload meets interactive latency requirements.
3. Adapter compatibility with the exact installed Diffusers revision.
4. Whether a trained LoRA improves UI-specific handoff consistency over base
   Qwen-Image-2.1.
5. Whether deterministic Layout IR remains the dominant bottleneck after
   Qwen migration.

## Implementation progress

| Slice | Status | Evidence |
|---|---|---|
| Single Qwen backend and mandatory image handoff | complete | `qwen_image21_backend.py` |
| No cross-model Polish fallback | complete | `SwapOrchestrator._load_with_recovery` |
| Final-render provenance | complete | `FinalizeOutput` and finalize/enhance flows |
| Adapter base/pipeline manifest writer and strict validation | complete | `qwen_manifest.py` and Qwen backend startup validation |
| Planner review source alignment | complete | `planner_review` critique/preference source |
| Official Qwen LoRA training entrypoint | pending | requires exact Diffusers training-script revision and GPU validation |
| Hardware VRAM/latency measurements | pending | target GPU run required |

This table is intentionally maintained with the implementation so the paper
does not present planned experiments as completed results.
