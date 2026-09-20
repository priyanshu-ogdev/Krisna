"""Utilities package: image manipulation, hashing, metrics, filesystem linking, and parquet exports."""

from __future__ import annotations

from data_forge.utils.hashing import hamming_distance, perceptual_hash, sha256_file
from data_forge.utils.image_utils import load_image, resize_for_model
from data_forge.utils.link_or_copy import link_or_copy
from data_forge.utils.parquet_writer import write_records_parquet

# Convenience aliases
hash_file = sha256_file
write_parquet_records = write_records_parquet

__all__ = [
    "load_image",
    "resize_for_model",
    "sha256_file",
    "hash_file",
    "perceptual_hash",
    "hamming_distance",
    "link_or_copy",
    "write_records_parquet",
    "write_parquet_records",
]
