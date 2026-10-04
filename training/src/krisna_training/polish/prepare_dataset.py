"""Builds a dataset directory in the standard format diffusers' dreambooth-
LoRA scripts (and DiffSynth-Studio) expect: a flat folder of images plus a
`metadata.jsonl` with `file_name`/`text` columns.

Accepts captions either as a pre-loaded dictionary {filename: caption}
or as a path to a .json / .jsonl file, parsing formats automatically.

UPGRADE (docs/review/34_model_review_3_polish_default.md): this pipeline
never used `source_caption` at all, even though data-forge's own
`_export_zimage` exports it specifically (its own comment: "a future
Z-Image fine-tune with caption conditioning would need source_caption to
apply DALL-E 3-style caption mixing the same way the sketch tier already
does"). `caption_mix_ratio` closes that gap. The SAME train/inference
distribution-matching concern that motivated Sketch's
caption_mix_ratio applies equally here: this tier is a diffusion image
generator that receives SHORT, casual user prompts at inference (via
polish_default_backend.py's effective_prompt), while every image here
was, until this fix, trained on ONLY its dense, VLM-recaptioned
description — zero exposure to short-caption style at all.

ARCHITECTURAL DIFFERENCE from Sketch's version, stated plainly: Sketch's
SketchTokenDataset re-rolls the dense-vs-source choice on every
`__getitem__` call (so a different training epoch can see a different
caption style for the same image). This tier trains through the
OFFICIAL, third-party `train_dreambooth_lora_z_image.py` script, which
has no hook for per-epoch/per-access randomization — `metadata.jsonl` is
written once, and whatever caption lands there is what every epoch of
that training run sees for that image. The mixing decision is therefore
made ONCE per image, at prepare() time, not per-epoch. This is a real,
weaker form of the same idea (less caption-style diversity across a
training run than Sketch gets), not a full equivalent — documented here
rather than silently presented as identical to Sketch's mechanism.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

log = logging.getLogger("krisna_training.polish.prepare_dataset")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_captions(
    captions_input: dict[str, str] | dict[str, dict] | str | Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse captions from dict, JSON file, or JSONL file.

    Returns (dense_captions, source_captions) — both {filename: caption},
    source_captions only populated when the input actually carries a
    `source_caption` field.

    The dict-input form now accepts TWO shapes per value, auto-detected:
    a plain string (`{"a.png": "a caption"}` — the original, simplest
    calling convention, no source_caption possible) or a nested dict
    (`{"a.png": {"caption": "...", "source_caption": "..."}}` — BUG FIX,
    docs/review/34_model_review_3_polish_default.md: sync_polish_default.py
    used to build a plain `{filename: caption_string}` dict itself,
    silently discarding source_caption before ever reaching this
    function — meaning this whole caption-mixing feature was completely
    inert for the real data-forge pipeline, the ONLY real production
    caller of this module, even though it worked correctly for the
    JSONL-file-path calling convention tested directly against this
    function. sync_polish_default.py was fixed to pass the richer nested
    shape instead of pre-flattening to strings.
    """
    if captions_input is None:
        return {}, {}
    if isinstance(captions_input, dict):
        dense: dict[str, str] = {}
        source: dict[str, str] = {}
        for fname, value in captions_input.items():
            if isinstance(value, dict):
                dense[fname] = value.get("caption", "")
                if value.get("source_caption"):
                    source[fname] = value["source_caption"]
            else:
                dense[fname] = value
        return dense, source

    cpath = Path(captions_input)
    if not cpath.exists():
        log.warning("captions_file_not_found", extra={"path": str(cpath)})
        return {}, {}

    raw = cpath.read_text(encoding="utf-8").strip()
    if not raw:
        return {}, {}

    captions: dict[str, str] = {}
    source_captions: dict[str, str] = {}
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
                name = Path(fname).name
                captions[name] = rec.get("caption", "")
                if rec.get("source_caption"):
                    source_captions[name] = rec["source_caption"]
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
                            name = Path(fname).name
                            captions[name] = item.get("caption", "")
                            if item.get("source_caption"):
                                source_captions[name] = item["source_caption"]
        except Exception as e:
            log.warning("captions_json_parse_failed", extra={"path": str(cpath), "error": str(e)})

    return captions, source_captions


def prepare(
    image_dir: str | Path,
    output_dir: str | Path,
    captions: dict[str, str] | str | Path | None = None,
    use_shared_instance_prompt: str | None = None,
    # UPGRADE (docs/review/34_model_review_3_polish_default.md): matches
    # Sketch tier's default exactly (SketchTokenDataset, TrainConfig,
    # all four sketch YAML configs) and the same Betker et al. 2023
    # (DALL-E 3) sweep finding that motivates it — see this module's
    # docstring for the full reasoning and the one real architectural
    # difference (resolved once per image here, not per-epoch).
    caption_mix_ratio: float = 0.95,
    seed: int | None = None,
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

    captions_dict, source_captions_dict = load_captions(captions)
    rng = random.Random(seed)  # seedable for reproducible dataset builds, unseeded (system entropy) by default

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    metadata_path = output_dir / "metadata.jsonl"
    written = 0
    mixed_to_source = 0
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for image_path in image_paths:
            dest = output_dir / image_path.name
            if not dest.exists() or dest.resolve() != image_path.resolve():
                shutil.copy2(image_path, dest)

            dense_caption = captions_dict.get(image_path.name)
            source_caption = source_captions_dict.get(image_path.name)
            if dense_caption is None:
                caption = use_shared_instance_prompt or ""
            elif source_caption and rng.random() >= caption_mix_ratio:
                caption = source_caption
                mixed_to_source += 1
            else:
                caption = dense_caption

            meta_file.write(json.dumps({"file_name": image_path.name, "text": caption}) + "\n")
            written += 1

    log.info(
        "polish_prepare_dataset_complete",
        extra={"written": written, "mixed_to_source_caption": mixed_to_source,
               "caption_mix_ratio": caption_mix_ratio},
    )
    return output_dir


def count_prepared(output_dir: str | Path) -> int:
    metadata_path = Path(output_dir) / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with metadata_path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# Alias for backward compatibility
prepare_dataset = prepare
