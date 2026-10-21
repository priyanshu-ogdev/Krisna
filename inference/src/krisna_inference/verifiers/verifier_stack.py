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
import math

log = logging.getLogger("krisna_inference (formerly krisna_orchestrator).verifiers.stack")

# CLIP ViT-L/14 (~900MB) + aesthetic linear head (~3KB) + NSFW ViT
# (~350MB) + easyocr detector+recognizer (~150MB) ≈ under 1.5GB, comfortably
# inside the "<1B combined, off-the-shelf" framing from §6's stack table.
VERIFIER_VRAM_RESERVED_GB = 1.5

# These are Krisna's explicit acceptance targets, not universal thresholds
# copied from the cited papers. CLIPScore and LPIPS establish useful metric
# families and human-correlation motivation, but neither gives a portable
# pass/fail cutoff for UI renders. A future calibration run must replace these
# targets with held-out baseline quantiles and confidence intervals.
EVALUATION_TARGETS = {
    "clip_alignment": 0.70,
    "ocr_readability": 0.80,
    "layout_iou": 0.70,
    "aesthetic": 0.70,
    "handoff_consistency": 0.65,
    "lock_preservation": 0.95,
    "seam_continuity": 0.85,
}
EVALUATION_POLICY_VERSION = "krisna.eval.v1"


class VerifierStack:
    def __init__(self) -> None:
        from krisna_inference.verifiers.aesthetic_safety import AestheticVerifier, SafetyVerifier
        from krisna_inference.verifiers.clip_alignment import CLIPAlignmentVerifier
        from krisna_inference.verifiers.handoff_consistency import HandoffConsistencyVerifier
        from krisna_inference.verifiers.layout_iou import LayoutIoUVerifier
        from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier
        from krisna_inference.verifiers.ocr_readability import OCRReadabilityVerifier

        self.clip_alignment = CLIPAlignmentVerifier()
        self.ocr_readability = OCRReadabilityVerifier()
        self.layout_iou = LayoutIoUVerifier()
        self.aesthetic = AestheticVerifier()
        self.safety = SafetyVerifier()
        self.handoff_consistency = HandoffConsistencyVerifier()
        self.lock_preservation = LockPreservationVerifier()

    def score_finalize_output(
        self,
        polished_image,
        prompt: str,
        sketch_image=None,
        expected_regions: list[dict] | None = None,
        expected_texts: list[str] | None = None,
        locked_regions: list[dict] | None = None,
        lock_donor_image=None,
        donor_map: dict | None = None,
    ) -> dict:
        """Returns a dict with exactly design_state.VerifierScores' fields:
        clip_alignment, ocr_readability, layout_iou, aesthetic,
        handoff_consistency, lock_preservation. Any verifier that raises is logged and scored
        None rather than aborting the whole pass.
        """
        scores: dict = {}
        source_donor = lock_donor_image if lock_donor_image is not None else sketch_image

        for name, fn in (
            ("clip_alignment", lambda: self.clip_alignment.score(polished_image, prompt)),
            ("ocr_readability", lambda: self.ocr_readability.score(polished_image, expected_texts=expected_texts)["score"]),
            (
                "layout_iou",
                lambda: self.layout_iou.score(polished_image, expected_regions or []),
            ),
            ("aesthetic", lambda: self.aesthetic.score(polished_image)),
            ("handoff_consistency", lambda: self._handoff_score(polished_image, sketch_image)),
            ("lock_preservation", lambda: self.lock_preservation.score(source_donor, polished_image, locked_regions, donor_map=donor_map)),
            ("seam_continuity", lambda: self.lock_preservation.seam_score(polished_image, locked_regions)),
        ):
            try:
                scores[name] = fn()
            except Exception as e:
                log.warning("verifier_failed", extra={"verifier": name, "error": str(e)})
                scores[name] = None

        return scores

    def evaluate_finalize_output(
        self,
        polished_image,
        prompt: str,
        sketch_image=None,
        expected_regions: list[dict] | None = None,
        expected_texts: list[str] | None = None,
        locked_regions: list[dict] | None = None,
        safety_score: float | None = None,
        lock_donor_image=None,
        donor_map: dict | None = None,
    ) -> dict:
        """Return a reproducible multi-metric evaluation receipt.

        The receipt deliberately distinguishes measured values, unavailable
        dimensions, project targets, and calibration status. It is suitable
        for inference evidence and training run joins, but does not pretend
        that published metric correlations imply Krisna-specific thresholds.
        """
        metrics = self.score_finalize_output(
            polished_image,
            prompt=prompt,
            sketch_image=sketch_image,
            expected_regions=expected_regions,
            expected_texts=expected_texts,
            locked_regions=locked_regions,
            lock_donor_image=lock_donor_image,
            donor_map=donor_map,
        )
        available = {name: value for name, value in metrics.items() if value is not None}
        missing = sorted(set(EVALUATION_TARGETS) - set(available))
        per_metric = {
            name: {
                "value": metrics.get(name),
                "target": target,
                "passed": metrics.get(name) is not None and metrics[name] >= target,
                "available": metrics.get(name) is not None,
            }
            for name, target in EVALUATION_TARGETS.items()
        }
        aggregate = None
        if available:
            # Geometric mean penalizes a single weak dimension more than a
            # simple average, which is preferable for UI quality gates.
            aggregate = math.prod(max(0.0, min(1.0, value)) for value in available.values()) ** (1 / len(available))
        failed = [name for name, result in per_metric.items() if result["available"] and not result["passed"]]
        return {
            "policy": EVALUATION_POLICY_VERSION,
            "metrics": metrics,
            "per_metric": per_metric,
            "missing_metrics": missing,
            "failed_metrics": failed,
            "aggregate_geometric_mean": aggregate,
            "safety_score": safety_score,
            "safety_gate_passed": safety_score is None or safety_score >= 0.90,
            "status": "incomplete" if missing else ("fail" if failed else "pass"),
            "calibration": "project_target_not_published_universal_threshold",
            "references": {
                "clip_alignment": "CLIPScore, arXiv:2104.08718",
                "perceptual_similarity": "LPIPS, arXiv:1801.03924",
                "human_evaluation": "TASTE, arXiv:2605.20731",
            },
        }

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
