"""OCR readability verifier — for UI/graphic design, text that renders as
garbled glyphs (a known diffusion-model failure mode) is a hard quality
problem, not a subtle one. This scores how confidently detectable and
well-formed any rendered text is, using easyocr's own per-region confidence
plus a simple legibility heuristic (rejecting regions that are mostly
non-alphanumeric noise, which easyocr sometimes reports as low-confidence
"text" over pure texture).

If the render has NO text at all, this returns 1.0 (nothing illegible to
find) rather than 0.0 — a text-free design isn't a readability failure.
"""

from __future__ import annotations

import re

_ALPHANUMERIC_RE = re.compile(r"[A-Za-z0-9]")


class OCRReadabilityVerifier:
    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=_cuda_available())

    def score(self, image) -> dict:
        """Returns {"score": float, "text_regions": [...]} — the
        text_regions list is what gets stored back into
        DesignState.finalize_output / a future OCR-derived structure field,
        not just the bare score."""
        self._load()
        import numpy as np

        arr = np.array(image.convert("RGB"))
        results = self._reader.readtext(arr)  # [(bbox, text, confidence), ...]

        if not results:
            return {"score": 1.0, "text_regions": []}

        legible_scores = []
        text_regions = []
        for bbox, text, confidence in results:
            alnum_ratio = (
                sum(1 for c in text if _ALPHANUMERIC_RE.match(c)) / max(1, len(text))
            )
            legible = confidence >= self.min_confidence and alnum_ratio > 0.3
            legible_scores.append(1.0 if legible else 0.0)
            text_regions.append(
                {"bbox": bbox, "text": text, "confidence": float(confidence), "legible": legible}
            )

        score = sum(legible_scores) / len(legible_scores)
        return {"score": score, "text_regions": text_regions}


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
