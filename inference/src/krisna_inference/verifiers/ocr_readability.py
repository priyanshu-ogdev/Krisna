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


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


class OCRReadabilityVerifier:
    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=_cuda_available())

    def score(self, image, expected_texts: list[str] | None = None) -> dict:
        """Returns {"score": float, "text_regions": [...]} — the
        text_regions list is what gets stored back into
        DesignState.finalize_output / a future OCR-derived structure field,
        not just the bare score."""
        self._load()
        import numpy as np

        arr = np.array(image.convert("RGB"))
        results = self._reader.readtext(arr)  # [(bbox, text, confidence), ...]

        if not results:
            if expected_texts:
                return {"score": 0.0, "text_regions": [], "cer": 1.0, "wer": 1.0}
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
        receipt = {"score": score, "text_regions": text_regions}
        if expected_texts:
            expected = [str(text).strip() for text in expected_texts if str(text).strip()]
            recognized = [str(region["text"]).strip() for region in text_regions if str(region["text"]).strip()]
            expected_text = " ".join(expected).lower()
            recognized_text = " ".join(recognized).lower()
            cer = _edit_distance(expected_text, recognized_text) / max(1, len(expected_text))
            expected_words = expected_text.split()
            recognized_words = recognized_text.split()
            wer = _edit_distance(" ".join(expected_words), " ".join(recognized_words)) / max(1, len(expected_words))
            receipt.update({"cer": min(1.0, cer), "wer": min(1.0, wer)})
            receipt["score"] = max(0.0, min(1.0, 1.0 - cer)) * score
        return receipt


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
