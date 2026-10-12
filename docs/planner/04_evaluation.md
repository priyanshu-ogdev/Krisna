# Stage 04 — evaluation design

## Corpus

Build 200–500 JSONL cases covering:

- normal requests;
- iterative style/palette/layout changes;
- locked regions;
- contradictions and ambiguity;
- critique follow-ups;
- long history;
- malformed-output traps;
- retrieval paraphrases;
- screenshot Review cases where supported.

## Metrics

### Contract

- first-pass JSON and schema validity;
- retry rate;
- irrecoverable failure rate;
- invalid delta rate;
- wrong-stage rate.

### Product

- constraint extraction accuracy;
- locked-constraint preservation;
- contradiction handling;
- critique usefulness;
- human rating;
- visual defect recall for Review.

### Runtime

- cold load;
- warm latency;
- first token;
- tokens/second;
- peak VRAM/RAM;
- swap, sleep, and wake latency;
- process and memory leaks.

## Current implementation

`evaluate_planner.py` runs JSONL cases and records output, errors, and latency.
It is deliberately a receipt generator, not a fabricated quality score.

## Gate

The selected profile must meet the agreed quality tolerance against BF16 and
stay within measured hardware envelopes. Unit tests cannot establish this gate.
