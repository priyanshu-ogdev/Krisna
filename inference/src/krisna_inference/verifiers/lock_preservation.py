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
    def score(
        self,
        source_image,
        rendered_image,
        locked_regions: list[dict] | None = None,
        donor_map: dict[str, Any] | None = None,
    ) -> float | None:
        """Computes exact pixel preservation (1.0 = bitwise identical).
        
        Supports both localized multi-subject crop-and-paste preservation (AnyDoor / MS-Diffusion)
        and 1-to-1 canvas exact pixel preservation.
        """
        if rendered_image is None or not locked_regions:
            return None

        from PIL import Image
        rendered_arr = np.asarray(rendered_image.convert("RGB"), dtype=np.float32)
        height, width = rendered_arr.shape[:2]

        has_crop_locks = any(
            (isinstance(r, dict) and r.get("source_crop_bbox")) or getattr(r, "source_crop_bbox", None)
            for r in locked_regions
        )

        if not has_crop_locks:
            if source_image is None:
                return None
            if source_image.size != rendered_image.size:
                log.warning(
                    "Source image size %s does not match rendered image size %s; dropping cross-aspect lock verification.",
                    source_image.size, rendered_image.size
                )
                return None

        source_arr = np.asarray(source_image.convert("RGB"), dtype=np.float32) if (source_image and source_image.size == rendered_image.size) else None

        diffs: list[float] = []

        for region in locked_regions:
            bbox = region.get("bbox") if isinstance(region, dict) else getattr(region, "bbox", region)
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = [float(value) for value in bbox[:4]]
            x0 = max(0, min(width, int(round(x * width if x <= 1 else x))))
            y0 = max(0, min(height, int(round(y * height if y <= 1 else y))))
            x1 = max(x0, min(width, int(round((x + w) * width if x <= 1 and w <= 1 else x + w))))
            y1 = max(y0, min(height, int(round((y + h) * height if y <= 1 and h <= 1 else y + h))))
            if x1 <= x0 or y1 <= y0:
                continue

            src_asset_id = region.get("source_asset_id") if isinstance(region, dict) else getattr(region, "source_asset_id", None)
            src_crop = region.get("source_crop_bbox") if isinstance(region, dict) else getattr(region, "source_crop_bbox", None)

            if src_crop and len(src_crop) == 4:
                donor = None
                if donor_map and src_asset_id and src_asset_id in donor_map:
                    donor = donor_map[src_asset_id]
                elif donor_map and "default" in donor_map:
                    donor = donor_map["default"]
                elif donor_map and len(donor_map) == 1:
                    donor = next(iter(donor_map.values()))
                elif source_image is not None:
                    donor = source_image

                if donor is not None:
                    dw, dh = donor.size
                    sbx, sby, sbw, sbh = [float(v) for v in src_crop]
                    sx0 = max(0, min(dw - 1, round(sbx * dw if sbx <= 1.0 else sbx)))
                    sy0 = max(0, min(dh - 1, round(sby * dh if sby <= 1.0 else sby)))
                    sx1 = max(sx0 + 1, min(dw, round((sbx + sbw) * dw if sbx <= 1.0 and sbw <= 1.0 else sbx + sbw)))
                    sy1 = max(sy0 + 1, min(dh, round((sby + sbh) * dh if sby <= 1.0 and sbh <= 1.0 else sby + sbh)))

                    cropped = donor.crop((sx0, sy0, sx1, sy1)).convert("RGB")
                    resampled = cropped.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
                    patch_arr = np.asarray(resampled, dtype=np.float32)
                    diff = float(np.abs(patch_arr - rendered_arr[y0:y1, x0:x1]).mean())
                    diffs.append(diff)
                    continue

            if source_arr is not None:
                diff = float(np.abs(source_arr[y0:y1, x0:x1] - rendered_arr[y0:y1, x0:x1]).mean())
                diffs.append(diff)

        if not diffs:
            return None
        mae = float(np.mean(diffs)) / 255.0
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
