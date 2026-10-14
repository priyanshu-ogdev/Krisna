"""Protected-region pixel preservation metric.

This is intentionally separate from layout IoU: a render can preserve a box
while changing every pixel inside it. The metric is a normalized mean absolute
error complement over the requested locked regions.
"""

from __future__ import annotations

import numpy as np


class LockPreservationVerifier:
    def score(self, source_image, rendered_image, locked_regions: list[dict] | None = None) -> float | None:
        if source_image is None or not locked_regions:
            return None
        source = source_image.convert("RGB").resize(rendered_image.size)
        source_arr = np.asarray(source, dtype=np.float32)
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
