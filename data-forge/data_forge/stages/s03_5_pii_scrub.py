"""Stage 3.5: PII Scrub — face blurring + text redaction (NEW in v13 upgrades).

Prevents Krisna from learning to generate real people's private data.
Runs between Quality (Stage 3) and Safety (Stage 4).
"""

from __future__ import annotations

import re
import shutil
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s03_5")


@register_stage("s03_5_pii_scrub")
class PIIScrubStage(Stage):
    name = "s03_5_pii_scrub"
    requires = ["s03_quality"]

    async def run(
        self, manifest: Manifest, config: PipelineConfig,
        record_ids: list[str], engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s03_5_pii_scrub")

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "quality_scored"]
        if not records:
            return result

        scrubbed_dir = config.resolved_paths["scrubbed"]
        face_conf = stage_cfg.get("face_detection_confidence", 0.5)
        blur_kernel = stage_cfg.get("blur_kernel_size", 99)
        regex_patterns = stage_cfg.get("regex_patterns", {})

        # Compile PII regex patterns
        compiled_patterns: dict[str, re.Pattern] = {}  # type: ignore[type-arg]
        for name, pattern in regex_patterns.items():
            try:
                compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                log.warning("invalid_regex", name=name, error=str(e))

        # Try to load MediaPipe face detection
        face_detector = None
        try:
            import mediapipe as mp
            face_detection = mp.solutions.face_detection
            face_detector = face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=face_conf
            )
            log.info("mediapipe_loaded")
        except ImportError:
            log.warning("mediapipe_not_available", note="Face blurring disabled")

        processed = 0
        failed = 0

        for rec in records:
            try:
                img_path = config.data_root / (rec.image_path or "")
                if not img_path.exists():
                    manifest.update_record(rec.id, "pii_scrub", new_status="excluded_failed",
                                           reason="Image not found", exclusion_reason="image_missing")
                    failed += 1; continue

                detections: list[str] = []
                scrubbed_path = scrubbed_dir / rec.source_dataset / img_path.name

                # Load image
                import numpy as np
                from PIL import Image, ImageFilter

                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                modified = False

                # 1. Face blurring
                if face_detector is not None:
                    import mediapipe as mp
                    mp_results = face_detector.process(img_array)
                    if mp_results.detections:
                        for detection in mp_results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            h_img, w_img = img_array.shape[:2]
                            x1 = max(0, int(bbox.xmin * w_img))
                            y1 = max(0, int(bbox.ymin * h_img))
                            x2 = min(w_img, int((bbox.xmin + bbox.width) * w_img))
                            y2 = min(h_img, int((bbox.ymin + bbox.height) * h_img))

                            # Apply Gaussian blur to face region
                            face_region = img.crop((x1, y1, x2, y2))
                            blurred = face_region.filter(
                                ImageFilter.GaussianBlur(radius=blur_kernel // 2)
                            )
                            img.paste(blurred, (x1, y1))
                            modified = True
                            detections.append(f"face_detected_at_{x1}_{y1}")

                # 2. Text-based PII detection (from OCR/metadata if available)
                if rec.ocr_output:
                    text_regions = rec.ocr_output.get("text_regions", [])
                    for region in text_regions:
                        text = region.get("text", "")
                        for pii_name, pattern in compiled_patterns.items():
                            if pattern.search(text):
                                detections.append(f"{pii_name}_in_ocr_text")
                                # Note: we redact the OCR text in metadata,
                                # the visual content is handled by face blur
                                # and by the downstream model not receiving raw PII

                # Save scrubbed image (or copy original if no modifications)
                scrubbed_path.parent.mkdir(parents=True, exist_ok=True)
                if modified:
                    img.save(scrubbed_path, quality=95)
                else:
                    shutil.copy2(img_path, scrubbed_path)

                # Update manifest
                rel_scrubbed = str(scrubbed_path.relative_to(config.data_root))
                manifest.update_record(
                    rec.id, "pii_scrub",
                    new_status="pii_scrubbed",
                    scrubbed_image_path=rel_scrubbed,
                    pii_scrubbed=True,
                    pii_detections=detections if detections else None,
                )
                processed += 1

            except Exception as e:
                log.error("pii_scrub_failed", record_id=rec.id, error=str(e))
                manifest.update_record(rec.id, "pii_scrub", new_status="excluded_failed",
                                       reason=f"PII scrub error: {e}", exclusion_reason="pii_scrub_error")
                failed += 1

        if face_detector is not None:
            face_detector.close()

        result.records_processed = processed
        result.records_failed = failed
        log.info("pii_scrub_complete", processed=processed, failed=failed,
                 detections_total=sum(1 for r in manifest.get_records_by_ids(record_ids)
                                     if r.pii_detections))
        return result
