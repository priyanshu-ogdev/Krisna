# Planner model artifacts

Downloaded model files belong here, never in the Python package tree. The
recommended native-PC layout is:

```text
models/planner/
  *.gguf
  mmproj-*.gguf          # required for image-capable Review mode
  manifest.json          # source revision and SHA256
  eval.json              # local bake-off output, not a checkpoint
```

Use `scripts/inference/download_planner_models.py` with an exact publisher
filename. The downloader refuses accidental overwrites and records hashes.
Training outputs retain their existing subdirectories (`sketch_tier/`,
`polish_default_lora/`, and `dpo_checkpoints/`).
