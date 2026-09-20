"""Handoff-consistency verifier — §5's architecture diagram lists this as
its own verifier stack dimension: does the Polish tier's output actually
preserve what the Sketch tier committed to, rather than drifting into a
different design entirely?

Uses CLIP image-image cosine similarity between the Stage-1 sketch (decoded
to pixels) and the Stage-2 polished render. This deliberately does NOT use
a pixel-level metric (SSIM/LPIPS) — Stage 2 is expected to change fine
detail, texture, and resolution substantially (§5: "Stage 80/100 — detail,
texture, lighting, final resolution"); CLIP embedding similarity tracks
semantic/layout consistency without penalizing that expected detail shift.

Same rescale caveat as clip_alignment.py: raw CLIP image-image similarity
for genuinely-the-same-scene image pairs tends to run higher than
image-text similarity (~0.6-0.9 is typical for "same design, different
render fidelity"), so the affine bounds here are different from
CLIP_SIM_LOW/HIGH in clip_alignment.py — again a documented heuristic, not
a calibrated threshold.
"""

from __future__ import annotations

HANDOFF_SIM_LOW = 0.5
HANDOFF_SIM_HIGH = 0.9


class HandoffConsistencyVerifier:
    def __init__(self, embedder=None) -> None:
        from krisna_inference.verifiers.common import get_clip_embedder

        self.embedder = embedder or get_clip_embedder()

    def score(self, sketch_image, polished_image) -> float:
        self.embedder.load()
        raw = self.embedder.image_image_similarity(sketch_image, polished_image)
        normalized = (raw - HANDOFF_SIM_LOW) / (HANDOFF_SIM_HIGH - HANDOFF_SIM_LOW)
        return max(0.0, min(1.0, normalized))
