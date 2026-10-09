"""Stage 6: Structure — UI component tree / layout JSON extraction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from PIL import UnidentifiedImageError

from data_forge.config import PipelineConfig
from data_forge.data.schema_validator import SchemaValidator
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.image_utils import load_image

log = get_logger("stages.s06")


@register_stage("s06_structure")
class StructureStage(Stage):
    name = "s06_structure"
    requires: ClassVar[tuple[str, ...]] = ("s05_recaption",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s06_structure")
        max_retries = stage_cfg.get("max_retries_on_schema_fail", 1)

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "recaptioned"]
        if not records or engine is None:
            return result

        tier1 = Tier1Engine(engine, config)
        validator = SchemaValidator(config.schemas_dir)
        processed = failed = 0

        async def _process(rec):
            raw_path = rec.scrubbed_image_path or rec.image_path
            if not raw_path:
                return {"id": rec.id, "status": "excluded_failed", "reason": "No image path specified", "exclusion_reason": "image_missing"}
            p = Path(raw_path)
            img_path = p if p.is_absolute() else (config.data_root / p)
            if not img_path.is_file():
                return {"id": rec.id, "status": "excluded_failed", "reason": "Image missing", "exclusion_reason": "image_missing"}

            # Bypass UI extraction for non-UI general visual datasets (e.g. pd12m, cc12m)
            from data_forge.data.domain_tagger import _GENERAL_DESIGN_SOURCES
            if rec.domain == "general_design" or rec.source_dataset in _GENERAL_DESIGN_SOURCES:
                from data_forge.inference.structured_output import StructureOutput
                empty_structure = StructureOutput(
                    elements=[],
                    layout_type="freeform",
                    hierarchy_depth=0,
                    background_style="image",
                )
                return {"id": rec.id, "status": "structured", "out": empty_structure.model_dump()}
            for attempt in range(max_retries + 1):
                try:
                    structure_out = await tier1.extract_structure(img_path)
                    if structure_out is not None:
                        out_dict = structure_out.model_dump()
                        valid, errors = validator.validate_structure(out_dict)
                        if valid:
                            break
                        log.warning("structure_schema_invalid", record_id=rec.id, attempt=attempt, errors=errors[:3])
                        structure_out = None  # Retry
                except Exception as e:
                    log.warning("structure_extraction_error", record_id=rec.id, attempt=attempt, error=str(e))
                    structure_out = None

            if structure_out is None:
                return {"id": rec.id, "status": "excluded_failed", "reason": "Structure extraction failed after retries or timeout", "exclusion_reason": "structure_extraction_failed"}
            else:
                return {"id": rec.id, "status": "structured", "out": structure_out.model_dump()}

        batch_concurrency = stage_cfg.get("batch_size", 64)
        sem = asyncio.Semaphore(batch_concurrency)
        async def _bounded_process(rec):
            async with sem:
                return await _process(rec)

        tasks = [asyncio.create_task(_bounded_process(rec)) for rec in records]
        results = await asyncio.gather(*tasks)

        updates: list[dict[str, Any]] = []
        for res in results:
            status = res["status"]
            update_item: dict[str, Any] = {
                "id": res["id"],
                "new_status": status,
                "structure_output": res.get("out"),
            }
            if status == "structured":
                processed += 1
            else:
                update_item["reason"] = res.get("reason")
                update_item["exclusion_reason"] = res.get("exclusion_reason")
                failed += 1
            updates.append(update_item)

        if updates:
            manifest.bulk_update_records(updates, stage="structure")

        result.records_processed = processed
        result.records_failed = failed
        return result
