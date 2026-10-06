"""Stage 5-OCR: OCR text extraction specialist stage."""

from __future__ import annotations

from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult


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
        if not records or engine is None:
            return result

        from data_forge.inference.ocr import OCREngine

        ocr = OCREngine(engine, config)
        processed = 0

        for rec in records:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                continue

            try:
                from data_forge.utils.image_utils import load_image

                _ = load_image(img_path)
            except Exception:
                continue

            ocr_out = await ocr.extract_text(img_path)
            if ocr_out:
                manifest.update_record(
                    rec.id,
                    "ocr_enrichment",
                    ocr_output=ocr_out.model_dump(),
                )
                processed += 1

        result.records_processed = processed
        return result
