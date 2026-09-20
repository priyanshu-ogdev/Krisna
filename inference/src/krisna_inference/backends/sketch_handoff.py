"""Builds a real `handoff_hook` for flows.finalize() — the "VQ-decode ->
pixel image" half of §5.3's Finalize diagram, using training/sketch's
VQTokenizer (the same tokenizer sketch-tier training data was encoded
with). The "-> re-encode into renderer's native latent" half doesn't need
a separate step here: both polish backends (polish_default_backend.py,
polish_quality_backend.py) already just take a pixel-image blob ref and do
their own internal encoding as part of their normal forward pass.
"""

from __future__ import annotations

from typing import Callable


def make_vq_decode_handoff(tokenizer, grid_h: int, grid_w: int) -> Callable[[list], str]:
    """tokenizer: a training.sketch.vq_tokenizer.VQTokenizer (already
    loaded or lazy — .decode() calls .load() itself if needed).
    Returns a callable suitable for flows.finalize(handoff_hook=...):
    takes the blob:// ref string from DesignState.sketch_tokens.vq_tokens,
    returns a blob:// ref to the decoded pixel image.
    """

    def handoff_hook(vq_tokens_ref: str) -> str:
        from krisna_inference.backends.blob_store_singleton import get_blob_store

        blobs = get_blob_store()
        tokens = blobs.load_tokens(vq_tokens_ref)
        image = tokenizer.decode(tokens, grid_h, grid_w)
        return blobs.save_image(image, prefix="sketch_handoff")

    return handoff_hook
