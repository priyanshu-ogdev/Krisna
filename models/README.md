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
Downloaded base model weights (`Qwen/Qwen-Image-2.1`) are cached in `models/` or HF cache, while fine-tuned adapters are placed in `checkpoints/` (`qwen_image21_lora_a6000_max/` or `qwen_image21_lora/`). The Sketch tier is deterministic layout code and requires no weights.
