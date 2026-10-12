# Stage 06 — structured generation

## Backend policy

- llama.cpp: grammar/JSON-schema constrained generation;
- vLLM: `StructuredOutputsParams(json=...)`;
- Transformers: strict extraction plus Pydantic validation initially.

The parent always validates output even when the runtime claims constrained
generation. A runtime failure must never apply a partial delta.

## Design issue to resolve

The user-facing reply and structured delta are two logical outputs. The final
schema should represent both explicitly rather than relying on “natural text
plus trailing JSON”. Streaming and UI behavior should be tested with the full
wrapper schema.

## Gate

At least 99% schema-valid output after one retry on the evaluation corpus, with
zero invalid deltas applied to persisted state.
