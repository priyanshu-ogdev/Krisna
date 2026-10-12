# Stage 01 — Planner contract

## Contract

The model/runtime boundary must return a validated product response:

```json
{
  "schema_version": "planner.response.v1",
  "reply_text": "...",
  "design_state_delta": {
    "stage": "conversing",
    "constraint_updates": {},
    "tool_call": null,
    "reasoning_note": ""
  },
  "retrieved_critique_ids": [],
  "runtime_metadata": {}
}
```

Only `design_state_delta` may update persisted state. Invalid output applies no
partial update and produces a typed Planner error.

## Current implementation

`planner_schema.py` validates stage, field types, bounded notes, and unknown
top-level delta fields. `planner_backend.py` validates the extracted object
before returning it.

## Required hardening

- validate constraint values against the Design State schema;
- define whether unknown constraint keys are allowed;
- define tool-call names and argument schemas;
- separate user-visible reply text from the JSON delta;
- add schema-version migration tests;
- attach runtime and token accounting consistently.

## Gate

Contract tests must cover valid output, malformed JSON, unknown fields, invalid
stage, oversized note, missing fields, and no-state-change-on-failure.
