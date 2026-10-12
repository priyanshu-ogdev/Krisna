# Training plan

## Phase A — IR and deterministic renderer

Build the schema, canonicalizer, legality checker, partial-mask generator, and wireframe renderer before training. Unit-test exact round trips.

## Phase B — Planner baseline

Use the frozen Planner with LayoutCoT-inspired retrieval and constrained JSON to generate layouts. This establishes a zero-training quality/latency ceiling.

## Phase C — LayoutFlow adaptation

Start from the official LayoutFlow architecture only as a reference; train a Krisna-specific model on the Krisna IR. Use masked flow matching for:

- full generation;
- known element types;
- known boxes;
- completion;
- locked-region preservation;
- refinement of noisy or user-edited layouts.

Losses:

```text
flow matching velocity loss
+ geometry legality penalty
+ relation consistency penalty
+ optional alignment/spacing regularizer
```

Do not use generated screenshots as the primary target.

## Phase D — optional visual tokenizer

Only if the wireframe-to-Polish handoff loses useful visual information, test TA-TiTok-style compact tokens or retain the existing MaskGIT baseline. Do not train a 568M/1.1B MaskGen model on the 12GB target without a separate hardware decision.
