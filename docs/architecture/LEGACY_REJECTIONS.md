# Superseded designs and release decisions

The first release targets 12GB VRAM and 16GB system RAM. These are deployment
requirements, not measured performance claims. This record preserves design
history; it does not authorize retaining alternate production implementations.

| Previous process | Decision and reason | Replacement | Evidence still required |
|---|---|---|---|
| Planner BF16 LoRA/SFT | Remove: Planner adaptation is frozen inference; no validated training corpus or adapter deployment contract | Prompting, retrieval, schema validation | Held-out planning accuracy |
| Transformers NF4 Planner | Remove: an additional runtime and unmeasured quantization policy complicate one consumer release | Pinned llama.cpp/GGUF | Tensor precision manifest and task accuracy |
| vLLM worker and sleep/wake | Remove from this release: Linux-oriented deployment and separate runtime stack; RAM parking has not been budgeted on 16GB hosts | Single llama.cpp lifecycle | Cold/warm latency and combined process RAM |
| Gemma 31B Critic | Remove: separate weights and process exceed the intended combined budget; CPU offload shifts pressure to RAM | Planner Review with image/evidence input | Human-scored defect recall; replacement is not yet proven equivalent |
| MaskGIT + VQGAN | Remove: discrete pixels do not encode IDs, hierarchy, exact text or hard locks | Structured Layout IR and renderer | Final-render layout fidelity and leakage tests |
| Flat LayoutFlow as a direct replacement | Reject unchanged: published flat, small-element layouts are not dense hierarchical screens | Parent-conditioned hierarchical layout research | Checkpoint parity, group-held-out evaluation |
| MaskGen as default Sketch | Reject: compact tokens do not imply a compact generator or local edit guarantees | Structured geometry | No implementation planned for release |
| AI-judge preference labels | Reject under project data policy | Human/public annotations with provenance | Dataset licenses and label-quality audit |
| Image-FID used as layout proof | Reject: feature networks and splits differ | Constraint metrics and properly specified layout-feature metrics | Evaluation protocol and uncertainty estimates |
| Z-Image-Turbo Polish | Remove: flow-matching assumptions, Turbo distillation, and custom DPO loss do not match the selected Qwen image-edit training contract | Qwen-Image-2.1 image-conditioned LoRA | Target-GPU benchmark and paired edit evaluation |
| Qwen-Image-Edit-2511 Polish | Remove: separate edit pipeline duplicates the renderer and prevents one consistent model/adapter provenance path | One Qwen-Image-2.1 pipeline for both Polish modes | Base-vs-LoRA human preference study |
| Flow-matching Diffusion-DPO | Remove: loss target and transformer API were architecture-specific to retired Z-Image | Official Qwen img2img DreamBooth LoRA | Preference-pair experiment designed for Qwen conditioning |

## What the historical pipeline actually did

The Planner emitted text followed by a JSON delta. Its Transformers backend
loaded NF4 weights; an optional vLLM process repeated similar generation logic.
Sketch encoded screenshots through VQGAN and predicted masked image tokens.
The sampler used argmax and confidence ranking; Halton order only broke ties,
and committed tokens were not remasked. Finalize decoded tokens back to pixels.
Critique unloaded the baseline, loaded Gemma, and restored the baseline later.

The new contract must preserve conversation, exact layout state, rendered
artifacts, and review results without either token decoding or a second critic.
Deleting files alone does not establish those producer/consumer contracts.

## Research and limitations

See [citations](../citations.md). LayoutFlow supports the small conditioned
geometry approach; LayoutCoT motivates a retrieval-conditioned baseline; neither
establishes quality on local quantized Qwen3.5. MaskGen results are image results,
not proof of locked-node editing. vLLM exclusion is a release-scope decision, not
a claim that vLLM cannot serve other small quantized models. Quantization quality
and 12GB/16GB feasibility require actual measurements, including the image
projector, context/recurrent state, renderer, operating system and model loading
peaks. Weight-file size is not a system-memory budget.

## Migration status

Runtime files for Gemma, MaskGIT, vLLM Planner, Z-Image Polish, Qwen-Edit
Polish, and the custom flow-matching DPO trainer have been deleted. The
obsolete DPO launcher/autodetection and REPA integration were also removed;
they depended on the retired trainer and could not produce Qwen-compatible
image-conditioned adapters. The
factory selects llama.cpp, LayoutBackend, and one Qwen-Image-2.1 backend. Full cleanup is still incomplete:
tests, setup/download scripts, legacy Sketch training, state-machine vocabulary,
and the token-based finalize handoff require migration. The layout backend is
currently a fixed template, not a trained generator. Review currently lacks a
verified image-processing path. These are blocking implementation gaps.
