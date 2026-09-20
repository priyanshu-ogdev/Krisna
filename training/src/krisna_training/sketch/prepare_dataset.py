"""Builds a training manifest from a directory of images (+ optional
captions.json mapping filename -> caption string) by tokenizing each image
through VQTokenizer and writing per-image .npy token grids plus a JSONL
manifest referencing them.

Usage: point `image_dir` at UI screenshots — e.g. RICO/CLAY/Enrico, the
same datasets data-forge already registers and downloads (see that
project's configs/datasets.yaml) — this doesn't consume data-forge's
manifest directly (see vq_tokenizer.py's docstring on why), it just
expects a flat directory of images as input, however you got them there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("krisna_training.sketch.prepare_dataset")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def prepare(
    image_dir: str | Path,
    output_dir: str | Path,
    tokenizer,
    image_size: int = 256,
    captions_path: str | Path | None = None,
) -> Path:
    """Returns the path to the written manifest.jsonl."""
    from PIL import Image

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[str, str] = {}
    if captions_path and Path(captions_path).exists():
        cpath = Path(captions_path)
        raw_text = cpath.read_text(encoding="utf-8").strip()
        if raw_text:
            if cpath.suffix.lower() == ".jsonl" or ("\n" in raw_text and not raw_text.startswith("[")):
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    fname = rec.get("image_filename") or rec.get("filename") or rec.get("source_image") or rec.get("record_id")
                    if fname:
                        captions[Path(fname).name] = rec.get("caption", "")
            else:
                try:
                    loaded = json.loads(raw_text)
                    if isinstance(loaded, dict):
                        captions = loaded
                    elif isinstance(loaded, list):
                        for item in loaded:
                            if isinstance(item, dict):
                                fname = item.get("image_filename") or item.get("filename") or item.get("source_image")
                                if fname:
                                    captions[Path(fname).name] = item.get("caption", "")
                except Exception as e:
                    log.warning("captions_load_failed", extra={"path": str(cpath), "error": str(e)})

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    grid_h, grid_w = tokenizer.grid_shape_for(image_size)
    manifest_path = output_dir / "manifest.jsonl"

    written = 0
    with manifest_path.open("w") as manifest_file:
        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                tokens = tokenizer.encode(image)
            except Exception as e:
                log.warning("tokenize_failed", extra={"path": str(image_path), "error": str(e)})
                continue

            import numpy as np

            token_file = f"tokens/{i:07d}.npy"
            np.save(tokens_dir / f"{i:07d}.npy", np.array(tokens, dtype="int32"))

            record = {
                "tokens_path": token_file,
                "caption": captions.get(image_path.name, ""),
                "source_image": str(image_path),
            }
            manifest_file.write(json.dumps(record) + "\n")
            written += 1

    log.info(
        "prepare_dataset_complete",
        extra={"written": written, "skipped": len(image_paths) - written, "grid": f"{grid_h}x{grid_w}"},
    )
    return manifest_path
