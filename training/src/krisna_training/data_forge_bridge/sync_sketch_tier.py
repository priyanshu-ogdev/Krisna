"""Reads data-forge's `model_data/sketch_tier_maskgit/{images/,captions.jsonl}`
and produces the JSONL manifest `training/sketch/dataset.py`'s
`SketchTokenDataset` expects — tokenizing each image via THIS project's
own `VQTokenizer`, not data-forge's `.pt` files. See this package's
`__init__.py` for why.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("krisna_training.data_forge_bridge.sync_sketch_tier")


def sync(
    data_forge_model_data_dir: str | Path,
    output_dir: str | Path,
    tokenizer,
    image_size: int = 256,
    expected_codebook_size: int | None = 16384,
) -> Path:
    """data_forge_model_data_dir: path to data-forge's `model_data/`
    (the parent of `sketch_tier_maskgit/`). tokenizer: a
    training.sketch.vq_tokenizer.VQTokenizer (already constructed, load()
    called lazily as needed). Returns the path to the written manifest.jsonl,
    in the exact shape training/sketch/prepare_dataset.py's own output uses
    (tokens_path + caption), so SketchTokenDataset needs no changes to
    consume it.
    """
    from PIL import Image

    # C2 validation: protect against out-of-vocabulary corruption if an alternative
    # VQGAN codebook size is supplied (e.g. 8192 vs expected MaskGIT 16384).
    if expected_codebook_size is not None:
        actual_size = getattr(tokenizer, "codebook_size", None)
        if callable(actual_size):
            actual_size = actual_size()
        if actual_size is not None and actual_size != expected_codebook_size:
            raise ValueError(
                f"VQTokenizer codebook size ({actual_size}) does not match "
                f"expected MaskGIT vocab_size ({expected_codebook_size}). "
                "Using mismatched codebooks causes silent out-of-vocabulary corruption."
            )

    sketch_dir = Path(data_forge_model_data_dir) / "sketch_tier_maskgit"
    images_dir = sketch_dir / "images"
    captions_path = sketch_dir / "captions.jsonl"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"{images_dir} not found — this needs data-forge's s12_model_data_export "
            "fix that links raw images (not just vq_tokens/) into sketch_tier_maskgit/. "
            "If you're on an older data-forge export, re-run stage s12."
        )

    # BUG FIX: this used to build captions_by_record keyed by record_id
    # and then look records up by `image_path.stem`, assuming a linked
    # image's filename stem equals its record_id. That's false for every
    # real data-forge export — record_id is a random uuid4 (see
    # data-forge's manifest.py::create_record), completely unrelated to
    # the scrubbed image's on-disk filename (named after the original
    # fetch-time file, e.g. "rico_core_0000042.png"). The lookup silently
    # missed on every record and produced empty captions for the entire
    # synced dataset with no error anywhere. data-forge's
    # s12_model_data_export.py now emits `image_filename` directly in
    # each captions.jsonl entry — this is the real join key, not a
    # filename guess. Older data-forge exports (pre-fix, no
    # `image_filename` field) fall back to the old stem-based behavior
    # with a loud warning, rather than silently producing empty captions
    # again on a stale export.
    captions_by_filename: dict[str, dict] = {}
    legacy_captions_by_record: dict[str, str] = {}
    saw_legacy_entry = False
    if captions_path.exists():
        with captions_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "image_filename" in rec and rec["image_filename"]:
                    captions_by_filename[rec["image_filename"]] = {
                        "caption": rec.get("caption", ""),
                        # Added alongside the caption-mixing fix (see
                        # dataset.py) — the original, short, human/
                        # source-dataset caption, carried through so it
                        # can be mixed in during training rather than
                        # discarded after only being used as a prompt
                        # hint inside data-forge's s05_recaption.py.
                        "source_caption": rec.get("source_caption"),
                    }
                else:
                    saw_legacy_entry = True
                    legacy_captions_by_record[rec["record_id"]] = rec.get("caption", "")

    if saw_legacy_entry and not captions_by_filename:
        log.warning(
            "sync_sketch_tier_legacy_captions_format",
            extra={
                "note": "captions.jsonl has no image_filename field — this is a "
                         "pre-fix data-forge export. Falling back to the old "
                         "(broken-for-real-exports) stem==record_id matching, which "
                         "will most likely produce empty captions for every record. "
                         "Re-run data-forge's s12_model_data_export stage on an "
                         "up-to-date data-forge checkout to get real captions.",
            },
        )

    output_dir = Path(output_dir)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    image_paths = sorted(images_dir.iterdir())
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")

    grid_h, grid_w = tokenizer.grid_shape_for(image_size)
    written = 0
    matched_captions = 0
    with manifest_path.open("w") as manifest_file:
        for i, image_path in enumerate(image_paths):
            source_caption = None
            if image_path.name in captions_by_filename:
                entry = captions_by_filename[image_path.name]
                caption = entry["caption"]
                source_caption = entry.get("source_caption")
                matched_captions += 1
            else:
                # Legacy fallback only — see the warning above. Kept so a
                # pre-fix export doesn't hard-crash, just silently (now
                # loudly, via the warning above) loses captions the same
                # way it always did.
                caption = legacy_captions_by_record.get(image_path.stem, "")

            try:
                image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                tokens = tokenizer.encode(image)
            except Exception as e:
                log.warning("resync_tokenize_failed", extra={"path": str(image_path), "error": str(e)})
                continue

            import numpy as np

            token_file = f"tokens/{i:07d}.npy"
            np.save(tokens_dir / f"{i:07d}.npy", np.array(tokens, dtype="int32"))
            manifest_file.write(
                json.dumps({
                    "tokens_path": token_file, "caption": caption,
                    "source_caption": source_caption, "source_image": str(image_path),
                }) + "\n"
            )
            written += 1

    log.info(
        "sync_sketch_tier_complete",
        extra={
            "written": written, "skipped": len(image_paths) - written,
            "grid": f"{grid_h}x{grid_w}", "matched_captions": matched_captions,
        },
    )
    return manifest_path
