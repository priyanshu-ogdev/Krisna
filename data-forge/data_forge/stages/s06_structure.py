"""Stage 6: Structure — UI component tree / layout JSON extraction."""

from __future__ import annotations

from typing import Any

from data_forge.config import PipelineConfig
from data_forge.data.schema_validator import SchemaValidator
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s06")


@register_stage("s06_structure")
class StructureStage(Stage):
    name = "s06_structure"
    requires = ["s05_recaption"]

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

        for rec in records:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                manifest.update_record(rec.id, "structure", new_status="excluded_failed",
                                       reason="Image missing", exclusion_reason="image_missing")
                failed += 1; continue

            structure_out = None
            for attempt in range(max_retries + 1):
                structure_out = await tier1.extract_structure(img_path)
                if structure_out is not None:
                    out_dict = structure_out.model_dump()
                    valid, errors = validator.validate_structure(out_dict)
                    if valid:
                        break
                    log.warning("structure_schema_invalid", record_id=rec.id,
                                attempt=attempt, errors=errors[:3])
                    structure_out = None  # Retry

            if structure_out is None:
                manifest.update_record(rec.id, "structure", new_status="excluded_failed",
                                       reason="Structure extraction failed after retries",
                                       exclusion_reason="structure_extraction_failed")
                failed += 1
            else:
                manifest.update_record(rec.id, "structure", new_status="structured",
                                       structure_output=structure_out.model_dump())
                processed += 1

        result.records_processed = processed
        result.records_failed = failed
        return result
