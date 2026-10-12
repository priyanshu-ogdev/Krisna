# Stage 05 — quantization bake-off

## Candidates

1. BF16 reference.
2. Current bitsandbytes NF4.
3. Selective NF4 with sensitive modules protected.
4. llama.cpp dynamic Q4.
5. Q5 and Q6 dynamic profiles.
6. Q8 Planner-alone ceiling.
7. Qwen3.5-4B fallback.
8. Protected W4A16 vLLM candidate.

## Measurement requirements

For every candidate record:

- exact model revision;
- quantizer/tool versions;
- calibration corpus and seed;
- layer/module policy;
- disk size;
- peak VRAM and RAM;
- context length;
- generation quality and contract metrics.

The protected W4A16 candidate is documented at approximately 11 GB because
vision, embeddings, and linear-attention components remain higher precision.
It must not be advertised as a 6–7 GB profile.

## Calibration

Inference calibration may use real UICrit text and representative Planner
prompts. It must not produce labels, LoRA updates, or a trainable Planner
checkpoint.

## Gate

Choose the smallest candidate within BF16 quality tolerance and measured VRAM
margin. If no candidate passes, keep the reference/fallback mode visible rather
than silently lowering quality.
