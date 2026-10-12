# Inference contract

```text
Planner
  → Layout IR
  → legality/constraint projection
  → deterministic wireframe renderer
  → Sketch handoff image + structured IR
  → Z-Image-Turbo / Qwen-Image-Edit
  → verifier stack
```

The IR remains authoritative throughout the pipeline. Polish receives:

- wireframe/reference image;
- normalized layout IR;
- user prompt;
- locked regions;
- content/style tokens.

The pixel Sketch model remains a compatibility fallback until this contract is proven.
