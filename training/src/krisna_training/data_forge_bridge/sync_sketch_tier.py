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
    (tokens_shard + tokens_index + caption, via the shared
    TokenShardWriter — see that module's docstring for why this replaced
    the older one-.npy-file-per-image format), so SketchTokenDataset
    needs no changes to consume it.
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

    output_dir = Path(output_dir)
    tokens_dir = output_dir / "tokens"
    manifest_path = output_dir / "manifest.jsonl"

    image_paths = sorted(images_dir / name for name in captions_by_filename)
    missing_images = [path.name for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Caption allowlist references missing images: {missing_images[:5]}")
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")

    grid_h, grid_w = tokenizer.grid_shape_for(image_size)

    # UPGRADE (docs/review/33_model_review_2_data_training_connection.md):
    # was one .npy file per image — a real I/O bottleneck at PRD §8.3's
    # target corpus scale (100K-500K images). See
    # token_shard_writer.py's module docstring for the full reasoning.
    from krisna_training.sketch.token_shard_writer import TokenShardWriter

    writer = TokenShardWriter(tokens_dir, expected_token_len=grid_h * grid_w)
    written = 0
    matched_captions = 0
    with manifest_path.open("w") as manifest_file:
        for image_path in image_paths:
            source_caption = None
            if image_path.name in captions_by_filename:
                entry = captions_by_filename[image_path.name]
                caption = entry["caption"]
                source_caption = entry.get("source_caption")
                matched_captions += 1
            else:
                raise RuntimeError(f"Image escaped the caption allowlist: {image_path.name}")

            try:
                image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                tokens = tokenizer.encode(image)
            except Exception as e:
                log.warning("resync_tokenize_failed", extra={"path": str(image_path), "error": str(e)})
                continue

            shard_path, index_in_shard = writer.add(tokens)
            manifest_file.write(
                json.dumps({
                    "tokens_shard": shard_path, "tokens_index": index_in_shard,
                    "caption": caption,
                    "source_caption": source_caption, "source_image": str(image_path),
                }) + "\n"
            )
            written += 1
    writer.close()  # flush any remaining partial shard — see its own docstring

    log.info(
        "sync_sketch_tier_complete",
        extra={
            "written": written, "skipped": len(image_paths) - written,
            "grid": f"{grid_h}x{grid_w}", "matched_captions": matched_captions,
        },
    )
    return manifest_path
