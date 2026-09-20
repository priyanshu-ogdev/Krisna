"""VerifierStack — runs all five verifiers from §5's architecture diagram
and returns a dict matching design_state.VerifierScores' field names
exactly, so flows.py can assign it straight into
DesignState.finalize_output.verifier_scores without any renaming.

Loaded once, kept resident (see this package's __init__.py docstring for
why it doesn't participate in ResidencyState swaps) — VERIFIER_VRAM_RESERVED_GB
is a documented, declared estimate for logging/budgeting purposes, not
enforced by VRAMLedger (that ledger only tracks the swappable + baseline
tiers; see swap_orchestrator.py).
"""

from __future__ import annotations

import logging

log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).verifiers.stack")

# CLIP ViT-L/14 (~900MB) + aesthetic linear head (~3KB) + NSFW ViT
# (~350MB) + easyocr detector+recognizer (~150MB) ≈ under 1.5GB, comfortably
# inside the "<1B combined, off-the-shelf" framing from §6's stack table.
VERIFIER_VRAM_RESERVED_GB = 1.5


class VerifierStack:
    def __init__(self) -> None:
        from krisna_inference.verifiers.aesthetic_safety import AestheticVerifier, SafetyVerifier
        from krisna_inference.verifiers.clip_alignment import CLIPAlignmentVerifier
        from krisna_inference.verifiers.handoff_consistency import HandoffConsistencyVerifier
        from krisna_inference.verifiers.layout_iou import LayoutIoUVerifier
        from krisna_inference.verifiers.ocr_readability import OCRReadabilityVerifier

        self.clip_alignment = CLIPAlignmentVerifier()
        self.ocr_readability = OCRReadabilityVerifier()
        self.layout_iou = LayoutIoUVerifier()
        self.aesthetic = AestheticVerifier()
        self.safety = SafetyVerifier()
        self.handoff_consistency = HandoffConsistencyVerifier()

    def score_finalize_output(
        self,
        polished_image,
        prompt: str,
        sketch_image=None,
        expected_regions: list[dict] | None = None,
    ) -> dict:
        """Returns a dict with exactly design_state.VerifierScores' fields:
        clip_alignment, ocr_readability, layout_iou, aesthetic,
        handoff_consistency. Any verifier that raises is logged and scored
        None rather than aborting the whole pass — a broken OCR install
        shouldn't block finalize from reporting the four scores it CAN
        compute.
        """
        scores: dict = {}

        for name, fn in (
            ("clip_alignment", lambda: self.clip_alignment.score(polished_image, prompt)),
            ("ocr_readability", lambda: self.ocr_readability.score(polished_image)["score"]),
            (
                "layout_iou",
                lambda: self.layout_iou.score(polished_image, expected_regions or []),
            ),
            ("aesthetic", lambda: self.aesthetic.score(polished_image)),
            ("handoff_consistency", lambda: self._handoff_score(polished_image, sketch_image)),
        ):
            try:
                scores[name] = fn()
            except Exception as e:
                log.warning("verifier_failed", extra={"verifier": name, "error": str(e)})
                scores[name] = None

        return scores

    def _handoff_score(self, polished_image, sketch_image) -> float | None:
        if sketch_image is None:
            return None  # nothing to compare against — e.g. no VQ-decode wired yet
        return self.handoff_consistency.score(sketch_image, polished_image)

    def safety_gate(self, image, min_safety_score: float = 0.9) -> tuple[bool, float]:
        """Separate from score_finalize_output because safety is a GATE
        (pass/fail before the user ever sees the image), not just another
        number in the scorecard. Returns (passed, score)."""
        score = self.safety.score(image)
        return score >= min_safety_score, score


_singleton: VerifierStack | None = None


def get_verifier_stack() -> VerifierStack:
    global _singleton
    if _singleton is None:
        _singleton = VerifierStack()
    return _singleton
