# Unified evaluation protocol

Status: implementation-aligned metric contract, 2026-10-14

Krisna now emits a `krisna.eval.v1` receipt for verifier-backed finalization.
The receipt combines several dimensions, reports missing dimensions explicitly,
and never turns a paper-reported correlation into an uncalibrated production
pass/fail claim.

## Metric stack

| Dimension | Current implementation | Standard / research basis | Current target |
|---|---|---|---:|
| Safety | Safety verifier gate | Product safety requirement | ≥ 0.90 |
| Text/image compatibility | CLIP alignment | CLIPScore shows reference-free CLIP metrics can correlate with human judgments, but warns of domain weaknesses ([arXiv:2104.08718](https://arxiv.org/abs/2104.08718)) | ≥ 0.70 |
| OCR readability | OCR confidence + alphanumeric heuristic; optional CER/WER against Layout IR literals | CER/WER are interpretable text fidelity metrics; fixed OCR engine and preprocessing are required | ≥ 0.80 score; CER/WER recorded when references exist |
| Layout adherence | Contour-detected region IoU against Layout IR and lock regions | IoU is a structural metric, not an aesthetic metric; future semantic detector can replace contours | ≥ 0.70 |
| Aesthetic quality | Existing aesthetic verifier | Must be calibrated against held-out human ratings; no universal threshold is assumed | ≥ 0.70 |
| Handoff consistency | CLIP image-image similarity | Deep perceptual features outperform shallow pixel metrics in human similarity judgments (LPIPS, [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)) | ≥ 0.65 |
| Lock preservation | Normalized mean absolute error complement inside protected regions | Exact protected-pixel requirement is a Krisna product constraint | ≥ 0.95 |

## Aggregation

The receipt records each metric separately and computes a geometric mean over
available metrics. The geometric mean penalizes a single weak dimension more
than an arithmetic mean. A receipt is:

- `pass` only when all required metrics are available and meet targets;
- `fail` when all required metrics are present but at least one misses target;
- `incomplete` when any required metric is unavailable;
- safety remains a separate hard gate and is never averaged away.

The current thresholds are **Krisna project targets**, not claimed universal
standards. They must be calibrated on a fixed source-disjoint validation set
with bootstrap confidence intervals before being used as paper results.

## Evaluation receipt

`FinalizeOutput.evaluation_receipt` contains:

- policy version;
- raw metric values;
- per-metric target and pass status;
- missing and failed metric lists;
- geometric aggregate;
- safety score and gate result;
- calibration status;
- metric references.

The same receipt is available through the normal session response and therefore
can be joined to a training or inference run ID without scraping logs.

## Validation procedure

1. Freeze model, adapter, prompt set, seed policy, evaluator versions, and
   source-disjoint split manifest.
2. Run the deterministic Layout IR baseline and the Qwen candidate on the same
   inputs.
3. Record per-sample receipts, not only aggregate means.
4. Report mean, median, standard deviation, bootstrap 95% confidence interval,
   missing-metric rate, safety-gate failures, and OOM/fallback rate.
5. Use TASTE as held-out human evaluation; do not use its scores to tune the
   thresholds on the same report.
6. Report CER/WER only with the exact OCR engine, language, preprocessing, and
   reference literals recorded.

## Interpretation boundary

CLIPScore, LPIPS, IoU, OCR, and aesthetic scores measure different properties.
No single score establishes UI quality. A high CLIP score cannot compensate for
bad typography, a high IoU cannot prove visual appeal, and a low LPIPS distance
is not automatically desirable for an intentional edit. The combined receipt
exists to expose these tradeoffs rather than hide them in one number.
