# Stage 03 — model acquisition and placement

## Layout

```text
models/planner/
  planner-*.gguf
  mmproj-*.gguf
  manifest.json
  eval/
```

`download_planner_models.py` requires an exact filename and writes a manifest
with repository, revision, artifact names, and SHA256. It refuses accidental
overwrites.

## Selection policy

Keep candidate profiles separate:

- dynamic Q4/Q5/Q6 for native PCs;
- Q8 as a Planner-alone quality ceiling;
- protected W4A16 for vLLM/high-memory Linux;
- BF16/NF4 for reference measurements;
- Qwen3.5-4B as an explicit fallback.

File size is not a VRAM measurement. Record loaded weights, KV cache, projector,
allocator reserve, and peak generation memory on real target hardware.

## Gate

No artifact becomes the default until its manifest, load receipt, generation
receipt, and evaluation report are present under `models/planner/`.
