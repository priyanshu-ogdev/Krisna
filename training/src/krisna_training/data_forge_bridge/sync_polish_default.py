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

from krisna_training.polish.prepare_dataset import prepare

log = logging.getLogger("krisna_training.data_forge_bridge.sync_polish_default")


def sync(
    data_forge_model_data_dir: str | Path, output_dir: str | Path,
    caption_mix_ratio: float = 0.95,
) -> Path:
    """Returns the prepared output_dir (same return convention as
    dataset_prep.prepare()). caption_mix_ratio: forwarded straight to
    prepare() — see that function's docstring; kept as an explicit
    parameter here (not just relying on prepare()'s own default) so a
    caller of THIS function can override it without needing to know
    prepare() exists underneath.
    """
    zimage_dir = Path(data_forge_model_data_dir) / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    captions_path = zimage_dir / "captions.jsonl"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"{images_dir} not found — this needs data-forge's s12_model_data_export "
            "fix that links raw images (not just latents/) into polish_zimage_turbo/. "
            "If you're on an older data-forge export, re-run stage s12."
        )

    # captions.jsonl is an allowlist as well as the caption join. Requiring
    # its current format prevents orphaned files from earlier exports from
    # being silently added to the next training set.
    if not captions_path.is_file():
        raise FileNotFoundError(f"Refusing to train from an export without its caption allowlist: {captions_path}")

    captions_by_filename: dict[str, dict] = {}
    with captions_path.open(encoding="utf-8") as captions_file:
        for line_number, line in enumerate(captions_file, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            filename = rec.get("image_filename")
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in {".", ".."}
                or "\\" in filename
            ):
                raise ValueError(
                    f"Invalid image_filename in {captions_path}:{line_number}; "
                    "the current export format is required"
                )
            if filename in captions_by_filename:
                raise ValueError(f"Duplicate training image in caption allowlist: {filename}")
            captions_by_filename[filename] = {
                "caption": rec.get("caption", ""),
                "source_caption": rec.get("source_caption"),
            }
    if not captions_by_filename:
        raise ValueError(f"Refusing to train: caption allowlist is empty in {captions_path}")
    log.info("sync_polish_default_start", extra={"images": len(list(images_dir.iterdir()))})
    return prepare(
        images_dir, output_dir, captions=captions_by_filename, use_shared_instance_prompt="a UI design",
        caption_mix_ratio=caption_mix_ratio,
        image_allowlist=set(captions_by_filename),
    )
