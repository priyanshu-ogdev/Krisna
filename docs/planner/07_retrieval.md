# Stage 07 — UICrit retrieval

## Current state

The repository supports TF-IDF and opt-in dense embeddings. TF-IDF has a
documented semantic-paraphrase failure mode.

## Target

Prebuild an offline index:

```text
models/planner_rag/
  documents.jsonl
  embeddings.npy
  metadata.json
  manifest.json
```

Use hybrid ranking:

```text
dense similarity + lexical similarity + metadata/domain score
```

Add relevance thresholds, duplicate suppression, snippet budgets, retrieval
IDs, and per-request telemetry. Do not recompute document embeddings at every
Planner load.

## Gate

Hybrid retrieval must beat the current TF-IDF baseline on a real, reviewed
UICrit query set without exceeding the Planner prompt budget.
