"""Stage 3.5: PII Scrub — face blurring (NEW in v13 upgrades).

Prevents Krisna from learning to generate real people's private data.
Runs between Quality (Stage 3) and Safety (Stage 4).

Text-based PII redaction is handled separately by s05_5_pii_text_redact,
which runs after OCR extraction is actually available (see that module's
docstring for why it can't happen here).
"""

from __future__ import annotations

import shutil
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.pii_faces import blur_faces, load_face_detector

log = get_logger("stages.s03_5")


@register_stage("s03_5_pii_scrub")
class PIIScrubStage(Stage):
    name = "s03_5_pii_scrub"
    requires: ClassVar[tuple[str, ...]] = ("s03_quality",)

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

        # Try to load MediaPipe face detection (shared helper — see
        # utils/pii_faces.py, also used by s01_6_preference_pairs.py)
        face_detector = load_face_detector(min_confidence=face_conf)
        if face_detector is not None:
            log.info("mediapipe_loaded")
        else:
            log.warning("mediapipe_not_available", note="Face blurring disabled")

        processed = 0
        failed = 0

        for rec in records:
            try:
                img_path = config.data_root / (rec.image_path or "")
                if not img_path.exists():
                    manifest.update_record(rec.id, "pii_scrub", new_status="excluded_failed",
                                           reason="Image not found", exclusion_reason="image_missing")
                    failed += 1
                    continue

                scrubbed_path = scrubbed_dir / rec.source_dataset / img_path.name

                # Load image
                from PIL import Image

                img = Image.open(img_path).convert("RGB")

                # 1. Face blurring (shared helper — see utils/pii_faces.py)
                img, modified, detections = blur_faces(img, face_detector, blur_kernel)

                # NOTE: text-based PII detection (email/phone/SSN/credit-card
                # regexes) does NOT run here. OCR extraction (rec.ocr_output)
                # doesn't happen until Stage 5's OCR enrichment sub-stage,
                # which runs after this one — so rec.ocr_output is always
                # empty at this point. See s05_5_pii_text_redact, which runs
                # after OCR and does the real text-PII scrub + visual
                # redaction. This stage only handles face blurring.

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
