"""Stage 5-OCR: OCR text extraction specialist stage."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.inference.ocr import OCREngine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.path_safety import resolve_data_path

log = get_logger("stages.s05_ocr")


@register_stage("s05_ocr_enrichment")
class OCREnrichmentStage(Stage):
    """Sub-stage: OCR text extraction (runs in a separate model swap phase)."""

    name = "s05_ocr_enrichment"
    requires: ClassVar[tuple[str, ...]] = ("s06_structure",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        records = manifest.get_records_by_ids(record_ids)
        # By the time this phase runs, Stage 6 (Structure) has normally advanced records
        # to "structured" — or "recaptioned" if structure extraction was skipped/disabled.
        records = [r for r in records if r.status in ("structured", "recaptioned")]
        if not records:
            return result
        from data_forge.data.domain_tagger import _GENERAL_DESIGN_SOURCES

        needs_ocr = any(
            not (r.domain == "general_design" or r.source_dataset in _GENERAL_DESIGN_SOURCES)
            for r in records
        )
        if needs_ocr and engine is None:
            updates = [{
                "id": rec.id,
                "new_status": "excluded_failed",
                "reason": "OCR inference engine unavailable; refusing unscanned text",
                "exclusion_reason": "ocr_engine_unavailable",
            } for rec in records if not (rec.domain == "general_design" or rec.source_dataset in _GENERAL_DESIGN_SOURCES)]
            manifest.bulk_update_records(updates, stage="ocr_enrichment")
            result.records_failed = len(updates)
            records = [r for r in records if (r.domain == "general_design" or r.source_dataset in _GENERAL_DESIGN_SOURCES)]
            if not records:
                return result

        ocr = OCREngine(engine, config) if (needs_ocr and engine is not None) else None
        processed = failed = 0

        async def _process(rec):
            # Bypass OCR extraction for non-UI general visual datasets (e.g. pd12m, cc12m)
            from data_forge.data.domain_tagger import _GENERAL_DESIGN_SOURCES
            if rec.domain == "general_design" or rec.source_dataset in _GENERAL_DESIGN_SOURCES:
                from data_forge.inference.structured_output import OCROutput
                empty_ocr = OCROutput(
                    text_regions=[],
                    primary_language="en",
                    total_text_regions=0,
                    confidence=1.0,
                )
                return rec.id, empty_ocr, None
            try:
                raw_path = rec.scrubbed_image_path or rec.image_path
                if not raw_path:
                    return rec.id, None, "Image path is missing"
                img_path = resolve_data_path(config.data_root, raw_path)
                ocr_out = await ocr.extract_text(img_path)
            except Exception as e:
                return rec.id, None, f"OCR inference failed: {e}"
            if ocr_out:
                return rec.id, ocr_out, None
            return rec.id, None, "OCR returned no structured result"

        stage_cfg = config.get_stage("s05_ocr_enrichment")
        batch_concurrency = stage_cfg.get("batch_size", 64)
        sem = asyncio.Semaphore(batch_concurrency)
        async def _bounded_process(rec):
            async with sem:
                return await _process(rec)

        tasks = [asyncio.create_task(_bounded_process(rec)) for rec in records]
        results = await asyncio.gather(*tasks)

        updates: list[dict[str, Any]] = []
        for rec_id, ocr_out, error in results:
            if error:
                updates.append({
                    "id": rec_id,
                    "new_status": "excluded_failed",
                    "reason": error,
                    "exclusion_reason": "ocr_failed",
                })
                failed += 1
            elif ocr_out is not None:
                ocr_payload = ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out
                updates.append({"id": rec_id, "ocr_output": ocr_payload})
                processed += 1

        if updates:
            # Batch update to DB at the end
            manifest.bulk_update_records(updates, stage="ocr_enrichment")

        result.records_processed = processed
        result.records_failed = failed
        return result
