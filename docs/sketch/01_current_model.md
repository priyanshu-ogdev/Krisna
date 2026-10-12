# Current Sketch model

## Architecture

`training/src/krisna_training/sketch/model.py` defines a bidirectional Transformer over VQGAN tokens. A learned mask token occupies ID `vocab_size`; prompt conditioning is projected and prepended as one prefix position. Two heads produce token logits and a sigmoid Token-Critic score.

## Training

Stage 1 uses a 16x16 token grid at 256px. Stage 2 uses a 32x32 grid at 512px and interpolates learned positional embeddings from Stage 1. The default loss is masked cross entropy with label smoothing plus critic BCE.

## Strengths

- small and understandable;
- already connected to the inference sampler;
- confidence/reveal behavior is explicitly trained;
- progressive resolution and checkpoint compatibility exist;
- no new foundation-model dependency.

## Limitations

- VQGAN is not UI-domain tuned;
- 2-D codebook tokens are expensive at 32x32;
- pixel-level reconstruction is not a reliable representation of layout constraints;
- learned absolute positions complicate resolution changes;
- CLIP text prefix is not a typed Planner-to-Sketch contract;
- locked-region editing is not represented as a first-class condition;
- current layout quality is evaluated indirectly rather than with a UI structural benchmark;
- the sampler's Halton sequence is only a candidate ordering/tie-breaker: candidates are sorted by combined confidence, and currently masked tokens are never remasked.

## Decision

Keep this model as the baseline and compatibility path. Do not delete its checkpoint or sampler until a structured replacement passes the same end-to-end renderer metrics.
