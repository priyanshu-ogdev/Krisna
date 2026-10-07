"""Stage 3.5: PII Scrub — face blurring (NEW in v13 upgrades).

Prevents Krisna from learning to generate real people's private data.
Runs between Quality (Stage 3) and Safety (Stage 4).

Text-based PII redaction is handled separately by s05_5_pii_text_redact,
which runs after OCR extraction is actually available (see that module's
docstring for why it can't happen here).
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import threading
from typing import Any, ClassVar

from PIL import Image

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.link_or_copy import link_or_copy
from data_forge.utils.path_safety import resolve_data_path
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
        records = [r for r in records if r.status in ("quality_scored", "pii_scrubbed")]
        if not records:
            return result

        paths = config.paths.resolve(config.data_root) if config.paths else config.resolved_paths
        scrubbed_dir = paths["scrubbed"]
        face_conf = stage_cfg.get("face_detection_confidence", 0.5)
        blur_kernel = stage_cfg.get("blur_kernel_size", 99)

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(config.data_root))
            except ValueError:
                return str(p)

        # Check if mediapipe is available at all
        test_detector = load_face_detector(min_confidence=face_conf)
        has_mediapipe = test_detector is not None
        if has_mediapipe:
            test_detector.close()
            log.info("mediapipe_loaded")
        else:
            log.error("mediapipe_not_available", note="Failing closed; no unscanned image will be promoted")
            updates = [{
                "id": rec.id,
                "new_status": "excluded_failed",
                "reason": "Face detector unavailable; PII scan could not run",
                "exclusion_reason": "pii_detector_unavailable",
                "pii_scrubbed": False,
            } for rec in records]
            manifest.bulk_update_records(updates, stage="pii_scrub")
            result.records_failed = len(updates)
            return result

        processed = 0
        failed = 0

        thread_local = threading.local()
        worker_detectors: list[Any] = []
        detector_lock = threading.Lock()
        
        def _get_detector():
            if not hasattr(thread_local, "detector"):
                thread_local.detector = load_face_detector(min_confidence=face_conf)
                if thread_local.detector is not None:
                    with detector_lock:
                        worker_detectors.append(thread_local.detector)
            return thread_local.detector

        def _process(rec):
            try:
                if not rec.image_path:
                    return {"id": rec.id, "status": "excluded_failed", "reason": "No image path specified", "exclusion_reason": "image_missing"}
                img_path = resolve_data_path(config.data_root, rec.image_path)
                if not img_path.is_file():
                    return {"id": rec.id, "status": "excluded_failed", "reason": "Image not found", "exclusion_reason": "image_missing"}
                
                scrubbed_path = scrubbed_dir / rec.source_dataset / f"{rec.id}{img_path.suffix.lower()}"
                with Image.open(img_path) as source_image:
                    image_format = source_image.format
                    img = source_image.convert("RGB")
                detector = _get_detector()
                if detector is None:
                    raise RuntimeError("Face detector failed to initialize in worker")
                img, modified, detections = blur_faces(img, detector, blur_kernel)
                
                scrubbed_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = scrubbed_path.with_name(f".{scrubbed_path.name}.tmp")
                if modified:
                    if image_format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                        raise ValueError(f"Unsupported image format for face redaction: {image_format}")
                    save_options = {"quality": 95} if image_format == "JPEG" else {}
                    img.save(temp_path, format=image_format, **save_options)
                    os.replace(temp_path, scrubbed_path)
                else:
                    link_or_copy(img_path, temp_path)
                    os.replace(temp_path, scrubbed_path)
                
                return {"id": rec.id, "status": "pii_scrubbed", "scrubbed_path": _rel(scrubbed_path), "detections": detections}
            except Exception as e:
                return {"id": rec.id, "status": "error", "error": str(e)}

        max_workers = min(16, max(2, multiprocessing.cpu_count()))
        updates: list[dict[str, Any]] = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for res in executor.map(_process, records):
                    if res["status"] == "pii_scrubbed":
                        updates.append({
                            "id": res["id"],
                            "new_status": "pii_scrubbed",
                            "scrubbed_image_path": res["scrubbed_path"],
                            "pii_scrubbed": True,
                            "pii_detections": res["detections"] if res["detections"] else None,
                        })
                        processed += 1
                    elif res["status"] == "error":
                        log.error("pii_scrub_failed", record_id=res["id"], error=res["error"])
                        updates.append({
                            "id": res["id"],
                            "new_status": "excluded_failed",
                            "reason": f"PII scrub error: {res['error']}",
                            "exclusion_reason": "pii_scrub_error",
                            "pii_scrubbed": False,
                        })
                        failed += 1
                    else:
                        updates.append({
                            "id": res["id"],
                            "new_status": res["status"],
                            "reason": res.get("reason"),
                            "exclusion_reason": res.get("exclusion_reason"),
                            "pii_scrubbed": False,
                        })
                        failed += 1

                    if len(updates) >= 500:
                        manifest.bulk_update_records(updates, stage="pii_scrub")
                        updates.clear()
        finally:
            for detector in worker_detectors:
                detector.close()

        if updates:
            manifest.bulk_update_records(updates, stage="pii_scrub")

        result.records_processed = processed
        result.records_failed = failed
        log.info("pii_scrub_complete", processed=processed, failed=failed)
        return result
