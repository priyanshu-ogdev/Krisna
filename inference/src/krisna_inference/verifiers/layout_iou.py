"""Layout-IoU verifier — how well does the finished render's actual layout
match the expected regions (DesignState.constraints.locked_regions, and/or
Stage-1's structure output)?

Region *detection* on the final render needs some way to find bounding
boxes in an arbitrary image without a trained UI-element detector (there
isn't one in this stack — see the sketch tier's own "no checkpoint" note).
The default detector here uses OpenCV contour detection on edges, which is
a real, functional, off-the-shelf technique (matches PRD §6's "off-the-shelf,
minimal training" framing for the verifier stack) — it finds rectangular
UI-block-shaped regions reasonably well, but it is NOT a semantic UI
element detector. `detector` is a constructor parameter specifically so a
real trained layout detector can be swapped in later without touching
anything that calls this verifier.
"""

from __future__ import annotations

from typing import Callable


class LayoutIoUVerifier:
    def __init__(self, detector: Callable | None = None) -> None:
        self.detector = detector or self._contour_detector

    def score(self, image, expected_regions: list[dict]) -> float:
        """expected_regions: list of {"bbox": [x, y, w, h]} in normalized
        (0-1) coordinates, e.g. from DesignState.constraints.locked_regions.
        Returns mean best-match IoU across expected regions (1.0 if there
        are no expected regions to check — nothing to fail)."""
        if not expected_regions:
            return 1.0

        detected = self.detector(image)
        if not detected:
            return 0.0

        ious = []
        for expected in expected_regions:
            best = max((_iou(expected["bbox"], d) for d in detected), default=0.0)
            ious.append(best)
        return sum(ious) / len(ious)

    @staticmethod
    def _contour_detector(image) -> list[list[float]]:
        import cv2
        import numpy as np

        arr = np.array(image.convert("L"))
        h, w = arr.shape
        edges = cv2.Canny(arr, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area_frac = (cw * ch) / (w * h)
            if area_frac < 0.005:  # discard noise-sized contours
                continue
            boxes.append([x / w, y / h, cw / w, ch / h])
        return boxes


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """box = [x, y, w, h], normalized coordinates."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
