"""Builds a dataset directory in the standard format diffusers' dreambooth-
LoRA scripts (and DiffSynth-Studio) expect: a flat folder of images plus a
`metadata.jsonl` with `file_name`/`text` columns.

Accepts captions either as a pre-loaded dictionary {filename: caption}
or as a path to a .json / .jsonl file, parsing formats automatically.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("krisna_training.polish.prepare_dataset")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_captions(captions_input: dict[str, str] | str | Path | None) -> dict[str, str]:
    """Parse captions from dict, JSON file, or JSONL file into {filename: caption}."""
    if captions_input is None:
        return {}
    if isinstance(captions_input, dict):
        return captions_input

    cpath = Path(captions_input)
    if not cpath.exists():
        log.warning("captions_file_not_found", extra={"path": str(cpath)})
        return {}

    raw = cpath.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    captions: dict[str, str] = {}
    if cpath.suffix.lower() == ".jsonl" or ("\n" in raw and not raw.startswith("[")):
        for line in raw.splitlines():
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
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                captions = loaded
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict):
                        fname = item.get("image_filename") or item.get("filename") or item.get("source_image")
                        if fname:
                            captions[Path(fname).name] = item.get("caption", "")
        except Exception as e:
            log.warning("captions_json_parse_failed", extra={"path": str(cpath), "error": str(e)})

    return captions


def prepare(
    image_dir: str | Path,
    output_dir: str | Path,
    captions: dict[str, str] | str | Path | None = None,
    use_shared_instance_prompt: str | None = None,
) -> Path:
    """image_dir: source images. output_dir: where the prepared dataset is
    written (a flat copy of the images + metadata.jsonl). captions: either
    maps source filename -> caption text, or points to a JSON / JSONL file.
    Images without an entry fall back to `use_shared_instance_prompt` if given.

    Returns output_dir.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions_dict = load_captions(captions)

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    metadata_path = output_dir / "metadata.jsonl"
    written = 0
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for image_path in image_paths:
            dest = output_dir / image_path.name
            if not dest.exists() or dest.resolve() != image_path.resolve():
                shutil.copy2(image_path, dest)

            caption = captions_dict.get(image_path.name)
            if caption is None:
                caption = use_shared_instance_prompt or ""

            meta_file.write(json.dumps({"file_name": image_path.name, "text": caption}) + "\n")
            written += 1

    return output_dir


def count_prepared(output_dir: str | Path) -> int:
    metadata_path = Path(output_dir) / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with metadata_path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# Alias for backward compatibility
prepare_dataset = prepare
