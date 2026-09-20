"""Reads data-forge's `model_data/polish_zimage_turbo/{images/,captions.jsonl}`
and produces the flat image+metadata.jsonl directory
`training/polish/dataset_prep.py`'s `prepare()` (and, downstream, the
official `train_dreambooth_lora_z_image.py` script) expects.

Unlike the sketch tier, no re-encoding happens here — Z-Image-Turbo's own
training script computes its own latents from raw images internally, and
data-forge's `images/` are already scrubbed/deduped/quality-gated, so this
is a straight format bridge (record-keyed captions.jsonl -> filename-keyed
metadata.jsonl), not a re-processing step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from krisna_training.polish.dataset_prep import prepare

log = logging.getLogger("krisna_training.data_forge_bridge.sync_polish_default")


def sync(data_forge_model_data_dir: str | Path, output_dir: str | Path) -> Path:
    """Returns the prepared output_dir (same return convention as
    dataset_prep.prepare())."""
    zimage_dir = Path(data_forge_model_data_dir) / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    captions_path = zimage_dir / "captions.jsonl"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"{images_dir} not found — this needs data-forge's s12_model_data_export "
            "fix that links raw images (not just latents/) into polish_zimage_turbo/. "
            "If you're on an older data-forge export, re-run stage s12."
        )

    # BUG FIX: this used to build captions_by_record keyed by record_id
    # then join `p.stem in captions_by_record`, comparing filename stems
    # against uuid4 record_ids — an intersection that's essentially
    # always empty for a real export, since record_id is unrelated to
    # the linked image's on-disk filename. data-forge's
    # s12_model_data_export.py now emits `image_filename` directly per
    # captions.jsonl entry — join on that instead of guessing. Falls back
    # to the old (broken) behavior with a loud warning for a pre-fix
    # data-forge export, same convention as sync_sketch_tier.py.
    captions_by_filename: dict[str, str] = {}
    saw_legacy_entry = False
    if captions_path.exists():
        with captions_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("image_filename"):
                    captions_by_filename[rec["image_filename"]] = rec.get("caption", "")
                else:
                    saw_legacy_entry = True

    if saw_legacy_entry and not captions_by_filename:
        log.warning(
            "sync_polish_default_legacy_captions_format",
            extra={
                "note": "captions.jsonl has no image_filename field — this is a "
                         "pre-fix data-forge export and will most likely produce "
                         "empty captions for every image. Re-run data-forge's "
                         "s12_model_data_export stage on an up-to-date checkout.",
            },
        )
    # BUG FIX (caught by test_sync_falls_back_to_shared_prompt_when_caption_missing):
    # there used to be a `.setdefault(p.name, "")` backfill loop here for
    # every image not already in captions_by_filename. That's wrong —
    # dataset_prep.prepare() distinguishes "no caption for this file"
    # (key absent -> `.get()` returns None -> falls back to
    # use_shared_instance_prompt) from "caption is the empty string"
    # (key present -> no fallback, image gets a literal blank caption).
    # Inserting "" for every unmatched image silently defeated the
    # shared-instance-prompt fallback for any image with no real caption
    # — exactly backwards from the intended behavior. Just don't touch
    # captions_by_filename for images with no match; let prepare()'s own
    # None-check handle it.

    log.info("sync_polish_default_start", extra={"images": len(list(images_dir.iterdir()))})
    return prepare(
        images_dir, output_dir, captions=captions_by_filename, use_shared_instance_prompt="a UI design"
    )
