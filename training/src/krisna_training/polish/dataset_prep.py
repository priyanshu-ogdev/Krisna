"""Builds a dataset directory in the standard format diffusers' dreambooth-
LoRA scripts (and DiffSynth-Studio) expect: a flat folder of images plus a
`metadata.jsonl` with `file_name`/`text` columns — the standard HF
`imagefolder` convention used across most diffusers training examples.

Also supports the simpler classic single-concept DreamBooth style (one
shared `instance_prompt` for every image, no per-image captions) via
`use_shared_instance_prompt`, for scripts/versions that don't support
per-image captioning. Check the actual training script's `--help` output
for which one it accepts — this project has not verified the exact flag
support of `train_dreambooth_lora_z_image.py`'s current version against a
live install (see this package's __init__.py for why).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def prepare(
    image_dir: str | Path,
    output_dir: str | Path,
    captions: dict[str, str] | None = None,
    use_shared_instance_prompt: str | None = None,
) -> Path:
    """image_dir: source images. output_dir: where the prepared dataset is
    written (a flat copy of the images + metadata.jsonl). captions: maps
    source filename -> caption text; images without an entry fall back to
    `use_shared_instance_prompt` if given, else an empty caption.

    Returns output_dir.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = captions or {}

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    metadata_path = output_dir / "metadata.jsonl"
    written = 0
    with metadata_path.open("w") as meta_file:
        for image_path in image_paths:
            dest = output_dir / image_path.name
            shutil.copy2(image_path, dest)

            caption = captions.get(image_path.name)
            if caption is None:
                caption = use_shared_instance_prompt or ""

            meta_file.write(json.dumps({"file_name": image_path.name, "text": caption}) + "\n")
            written += 1

    return output_dir


def count_prepared(output_dir: str | Path) -> int:
    metadata_path = Path(output_dir) / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with metadata_path.open() as f:
        return sum(1 for line in f if line.strip())
