"""CLIP alignment verifier — does the rendered design actually match the
constraints/prompt it was supposed to satisfy?

Raw CLIP cosine similarity for ViT-L/14 typically falls in ~0.15 (unrelated)
to ~0.35 (strongly related) — NOT 0-1. The affine rescale below maps that
observed range onto DesignState's 0-1 `clip_alignment` score. The exact
bounds (0.15/0.35) are a documented heuristic, not a calibrated threshold —
tighten them against a labeled UI-design sample before using this score for
anything higher-stakes than a rough finalize-quality gate.
"""

from __future__ import annotations

CLIP_SIM_LOW = 0.15
CLIP_SIM_HIGH = 0.35


class CLIPAlignmentVerifier:
    def __init__(self, embedder=None) -> None:
        from krisna_inference.verifiers.common import get_clip_embedder

        self.embedder = embedder or get_clip_embedder()

    def score(self, image, prompt: str) -> float:
        self.embedder.load()
        raw = self.embedder.image_text_similarity(image, prompt)
        normalized = (raw - CLIP_SIM_LOW) / (CLIP_SIM_HIGH - CLIP_SIM_LOW)
        return max(0.0, min(1.0, normalized))
