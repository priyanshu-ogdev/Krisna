"""Protected-region pixel preservation and seam boundary continuity metric.

This is intentionally separate from layout IoU: a render can preserve a box
while changing every pixel inside it. The metrics evaluate:
1. Pixel preservation: normalized mean absolute error complement over requested locked regions.
2. Seam boundary continuity: gradient / color step across the 1-pixel perimeter between
   locked donor pixels and newly generated canvas pixels.
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger("krisna_inference.verifiers.lock_preservation")


class LockPreservationVerifier:
    def score(self, source_image, rendered_image, locked_regions: list[dict] | None = None) -> float | None:
        """Computes exact pixel preservation (1.0 = bitwise identical)."""
        if source_image is None or not locked_regions:
            return None
        if source_image.size != rendered_image.size:
            log.warning(
                "Source image size %s does not match rendered image size %s; dropping cross-aspect lock verification.",
                source_image.size, rendered_image.size
            )
            return None

        source_arr = np.asarray(source_image.convert("RGB"), dtype=np.float32)
        rendered_arr = np.asarray(rendered_image.convert("RGB"), dtype=np.float32)
        mask = np.zeros(rendered_arr.shape[:2], dtype=bool)
        height, width = mask.shape

        for region in locked_regions:
            bbox = region.get("bbox") if isinstance(region, dict) else region
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = [float(value) for value in bbox[:4]]
            x0 = max(0, min(width, int(round(x * width if x <= 1 else x))))
            y0 = max(0, min(height, int(round(y * height if y <= 1 else y))))
            x1 = max(x0, min(width, int(round((x + w) * width if x <= 1 and w <= 1 else x + w))))
            y1 = max(y0, min(height, int(round((y + h) * height if y <= 1 and h <= 1 else y + h))))
            mask[y0:y1, x0:x1] = True

        if not mask.any():
            return None
        mae = np.abs(source_arr - rendered_arr)[mask].mean() / 255.0
        return float(max(0.0, min(1.0, 1.0 - mae)))

    def seam_score(self, rendered_image, locked_regions: list[dict] | None = None) -> float | None:
        """Evaluates boundary continuity across the perimeter interface of locked regions.

        Returns:
            float in [0.0, 1.0], where 1.0 indicates continuous blending and 0.0 indicates a severe edge artifact.
        """
        if rendered_image is None or not locked_regions:
            return None

        arr = np.asarray(rendered_image.convert("RGB"), dtype=np.float32)
        height, width, _ = arr.shape
        diffs: list[float] = []

        for region in locked_regions:
            bbox = region.get("bbox") if isinstance(region, dict) else region
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = [float(value) for value in bbox[:4]]
            x0 = max(1, min(width - 2, int(round(x * width if x <= 1 else x))))
            y0 = max(1, min(height - 2, int(round(y * height if y <= 1 else y))))
            x1 = max(x0 + 1, min(width - 1, int(round((x + w) * width if x <= 1 and w <= 1 else x + w))))
            y1 = max(y0 + 1, min(height - 1, int(round((y + h) * height if y <= 1 and h <= 1 else y + h))))

            # Top seam: diff between y0-1 and y0
            if y0 > 0:
                top_diff = np.abs(arr[y0 - 1, x0:x1] - arr[y0, x0:x1]).mean()
                diffs.append(float(top_diff))
            # Bottom seam: diff between y1 and y1-1
            if y1 < height:
                bottom_diff = np.abs(arr[y1, x0:x1] - arr[y1 - 1, x0:x1]).mean()
                diffs.append(float(bottom_diff))
            # Left seam: diff between x0-1 and x0
            if x0 > 0:
                left_diff = np.abs(arr[y0:y1, x0 - 1] - arr[y0:y1, x0]).mean()
                diffs.append(float(left_diff))
            # Right seam: diff between x1 and x1-1
            if x1 < width:
                right_diff = np.abs(arr[y0:y1, x1] - arr[y0:y1, x1 - 1]).mean()
                diffs.append(float(right_diff))

        if not diffs:
            return 1.0
        avg_boundary_step = float(np.mean(diffs)) / 255.0
        # Typical UI edge contrast is ~0.15; score penalizes abrupt step discontinuities
        return float(max(0.0, min(1.0, 1.0 - (avg_boundary_step * 2.0))))
