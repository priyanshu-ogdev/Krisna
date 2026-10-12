# Evaluation

## Structured metrics

- node-type accuracy;
- box IoU and generalized IoU;
- parent/tree accuracy;
- relation accuracy;
- locked-node preservation;
- legality violation rate;
- completion quality;
- alignment and overlap;
- layout FID only within the same dataset/split/protocol.

## End-to-end metrics

Render the generated IR and pass it through the same Polish stack. Measure:

- OCR text accuracy;
- layout IoU against known references;
- component placement error;
- style/constraint adherence;
- image quality;
- human pairwise preference;
- time to first usable sketch;
- final render improvement over the current MaskGIT handoff.

## Required A/B matrix

1. current MaskGIT;
2. Planner JSON layout;
3. LayoutFlow-adapted IR model;
4. LayoutFlow + Planner retrieval;
5. optional TA-TiTok/MaskGen visual route.

No model replaces the current path until the end-to-end A/B shows a real improvement, not just a better layout-only FID.
