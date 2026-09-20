"""Content hashing — perceptual and cryptographic."""

from __future__ import annotations

import hashlib
from pathlib import Path

from data_forge.logging_setup import get_logger

log = get_logger("utils.hashing")


def sha256_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(image_path: Path, hash_size: int = 16) -> str:
    """Compute a perceptual hash (average hash) of an image.

    Returns a hex string. Images that look similar will have similar hashes.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        # Resize to hash_size x hash_size, grayscale
        img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())

    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)

    # Convert to hex
    hex_hash = hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)
    return hex_hash


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two hex hashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError("Hashes must be the same length")

    bin_a = bin(int(hash_a, 16))[2:].zfill(len(hash_a) * 4)
    bin_b = bin(int(hash_b, 16))[2:].zfill(len(hash_b) * 4)

    return sum(a != b for a, b in zip(bin_a, bin_b))
