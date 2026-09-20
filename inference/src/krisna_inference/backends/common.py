"""Shared helpers for the inference backends.

`BlobStore` gives `image_ref` / `vq_tokens` fields in DesignState a genuine
on-disk meaning: a `blob://<name>` string that resolves to a real file
under `krisna_blobs/`, instead of the placeholder strings the mock outputs
use. Every real backend that produces or consumes an image writes/reads
through this, so `finalize_output.image_ref` and the "critique this"
flow's read of it are actually the same file on disk.
"""

from __future__ import annotations

import uuid
from pathlib import Path


class BlobStore:
    def __init__(self, root: str | Path = "krisna_blobs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, prefix: str = "img") -> str:
        """image: a PIL.Image.Image. Returns a 'blob://<filename>' ref."""
        name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        image.save(self.root / name)
        return f"blob://{name}"

    def save_tokens(self, tokens: list[int], prefix: str = "tokens") -> str:
        """tokens: a flat list of VQ token ids. Returns a 'blob://<filename>'
        ref — same convention as save_image, so DesignState.sketch_tokens
        .vq_tokens (typed str | None per §5.1) stays a reference, not raw
        data, consistent with how image_ref works."""
        import json

        name = f"{prefix}_{uuid.uuid4().hex[:12]}.json"
        (self.root / name).write_text(json.dumps(tokens))
        return f"blob://{name}"

    def load_tokens(self, ref: str) -> list[int]:
        import json

        if not ref.startswith("blob://"):
            raise ValueError(f"Not a blob ref: {ref!r}")
        path = self.root / ref.removeprefix("blob://")
        if not path.exists():
            raise FileNotFoundError(f"Blob not found on disk: {path}")
        return json.loads(path.read_text())

    def load_image(self, ref: str):
        from PIL import Image

        if not ref.startswith("blob://"):
            raise ValueError(f"Not a blob ref: {ref!r}")
        path = self.root / ref.removeprefix("blob://")
        if not path.exists():
            raise FileNotFoundError(f"Blob not found on disk: {path}")
        return Image.open(path)

    def path_for(self, ref: str) -> Path:
        return self.root / ref.removeprefix("blob://")

    def delete(self, ref: str) -> bool:
        """Delete the file backing `ref` from disk if present.

        Returns True if deleted, False if file did not exist.
        Raises ValueError if ref is not a valid blob:// ref.
        """
        if not ref.startswith("blob://"):
            raise ValueError(f"Not a blob ref: {ref!r}")
        path = self.root / ref.removeprefix("blob://")
        if path.exists():
            try:
                path.unlink()
                return True
            except OSError:
                return False
        return False


def resolve_dtype(name: str):
    """Lazy torch dtype lookup — only imports torch when actually called."""
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype name: {name!r}")
    return mapping[name]


def pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
