# Layout data contract

## Sources

Use real public annotations where available:

- RICO for mobile UI structure;
- Enrico/WebUI and other existing UI sources in data-forge;
- data-forge structure JSON where it is a real extracted artifact;
- held-out TASTE/PartiPrompts only for evaluation.

Do not silently create AI-judge layout labels. If VLM-derived structure is ever introduced, it must be marked synthetic, isolated, and excluded from claims about human-labeled training signal.

## Preparation

1. canonicalize element types;
2. normalize coordinates;
3. repair or reject invalid trees;
4. split by application/page identity to prevent near-duplicate leakage;
5. create partial-layout masks for completion;
6. create locked-node masks from real available structure;
7. export `layout_train.jsonl`, `layout_val.jsonl`, and `layout_test.jsonl`;
8. store provenance and license metadata per record.

## No synthetic shortcut

Do not use blurred screenshots as a proxy for Sketch output. The structured route should train on actual structure or clearly identified human/public annotations.
