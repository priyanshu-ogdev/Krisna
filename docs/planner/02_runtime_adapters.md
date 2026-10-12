# Stage 02 — runtime adapters

## Target

Planner behavior is runtime-independent. The runtime only loads weights and
generates text or multimodal output.

```python
load() -> receipt
generate(messages, schema, generation_config) -> generation
unload() -> None
sleep(level) -> None
wake() -> None
health() -> receipt
```

## Runtime roles

### llama.cpp / GGUF

Default for native Windows and low-VRAM PCs. It requires an exact GGUF and,
for image Review, a matching multimodal projector. The current adapter is
optional-import and fails clearly when the artifact or dependency is missing.

### vLLM

Linux/WSL2 and larger GPUs. It remains isolated because its Torch/runtime pins
must not contaminate the Diffusers inference environment. The worker now has a
JSON-schema generation hook.

### Transformers

Reference path for BF16/NF4/selective quantization comparisons and emergency
fallbacks. It is not the preferred 12-GB production runtime.

## Risks

- llama.cpp's exact Qwen3.5 GGUF/mmproj must be tested, not inferred from
  generic multimodal support;
- native Windows CUDA wheels are platform-specific;
- vLLM is not natively supported on Windows;
- text-only planning and vision Review may require different processor paths.

## Gate

All enabled runtimes must return the same validated contract for the same case,
with a receipt identifying runtime, model, quantization, latency, and token
usage.
