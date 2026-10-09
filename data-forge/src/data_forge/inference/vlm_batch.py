"""VLM Unified Batch Processing & Pipeline Extraction Coordinator.

Provides high-performance data structures and concurrent record extraction
to eliminate redundant sequential barrier passes over the manifest, redundant
disk I/O, base64 encoding cycles, and Vision Transformer prefill recomputations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.data.schema_validator import SchemaValidator
from data_forge.inference.client import ImagePayloadCache, get_global_image_cache
from data_forge.inference.ocr import OCREngine, OCROutput
from data_forge.inference.structured_output import CaptionOutput, StructureOutput
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import ManifestRecord
from data_forge.utils.path_safety import resolve_data_path

log = get_logger("inference.vlm_batch")


@dataclass
class VLMRecordContext:
    """In-memory data structure for a record passing through VLM processing stages.

    Caches the loaded image payload and coordinates multi-stage VLM outputs
    (recaption, structural JSON, OCR) to eliminate redundant disk I/O, base64
    encoding, and ViT patch prefill recalculations.
    """

    record_id: str
    manifest_record: ManifestRecord
    image_path: Path | None = None
    source_caption: str | None = None

    # Cached encoded image payload (mime_type, base64_str)
    image_payload: tuple[str, str] | None = None

    # Stage outputs
    caption_output: CaptionOutput | None = None
    structure_output: StructureOutput | None = None
    ocr_output: Any | None = None

    # Execution state
    status: str = "pending"
    exclusion_reason: str | None = None
    failure_reason: str | None = None


class VLMUnifiedPassCoordinator:
    """Coordinates high-throughput, concurrent VLM extractions for record batches."""

    def __init__(
        self,
        config: PipelineConfig,
        tier1_engine: Tier1Engine,
        ocr_engine: OCREngine | None = None,
        image_cache: ImagePayloadCache | None = None,
    ) -> None:
        self.config = config
        self.tier1 = tier1_engine
        if ocr_engine is not None:
            self.ocr = ocr_engine
        elif hasattr(tier1_engine, "_engine") and tier1_engine._engine is not None:
            self.ocr = OCREngine(tier1_engine._engine, config)
        else:
            self.ocr = None
        self.image_cache = image_cache or get_global_image_cache()
        self.validator = SchemaValidator(config.schemas_dir)

        structure_cfg = config.get_stage("s06_structure")
        self.max_structure_retries = structure_cfg.get("max_retries_on_schema_fail", 1)

    async def _extract_structure_with_retry(
        self,
        ctx: VLMRecordContext,
    ) -> StructureOutput | None:
        """Extract UI structure with schema validation and retry loop."""
        if ctx.image_path is None:
            return None

        structure_out = None
        for attempt in range(self.max_structure_retries + 1):
            try:
                structure_out = await self.tier1.extract_structure(
                    ctx.image_path,
                    image_payload=ctx.image_payload,
                )
                if structure_out is not None:
                    out_dict = structure_out.model_dump()
                    valid, errors = self.validator.validate_structure(out_dict)
                    if valid:
                        return structure_out
                    log.warning(
                        "structure_schema_invalid",
                        record_id=ctx.record_id,
                        attempt=attempt,
                        errors=errors[:3],
                    )
                    structure_out = None
            except Exception as e:
                log.warning(
                    "structure_extraction_error",
                    record_id=ctx.record_id,
                    attempt=attempt,
                    error=str(e),
                )
                structure_out = None
        return None

    async def process_record(
        self,
        ctx: VLMRecordContext,
        run_recaption: bool = True,
        run_structure: bool = True,
        run_ocr: bool = True,
    ) -> VLMRecordContext:
        """Process a single record concurrently across requested VLM stages."""
        rec = ctx.manifest_record
        raw_path = rec.scrubbed_image_path or rec.image_path
        if not raw_path:
            ctx.status = "excluded_failed"
            ctx.failure_reason = "No image path specified"
            ctx.exclusion_reason = "image_missing"
            return ctx

        try:
            ctx.image_path = resolve_data_path(self.config.data_root, raw_path)
            if not ctx.image_path.is_file():
                ctx.status = "excluded_failed"
                ctx.failure_reason = "Image missing"
                ctx.exclusion_reason = "image_missing"
                return ctx
        except Exception as e:
            ctx.status = "excluded_failed"
            ctx.failure_reason = f"Image path resolution failed: {e}"
            ctx.exclusion_reason = "image_missing"
            return ctx

        # Pre-cache image payload once in worker thread (1 disk read, 1 base64 encode)
        try:
            ctx.image_payload = await asyncio.to_thread(
                self.image_cache.get_or_encode, ctx.image_path
            )
        except Exception as e:
            ctx.status = "excluded_failed"
            ctx.failure_reason = f"Image read/encode failed: {e}"
            ctx.exclusion_reason = "image_corrupt"
            return ctx

        ctx.source_caption = rec.source_caption

        # Build concurrent tasks
        async def _caption_task() -> CaptionOutput | None:
            if not run_recaption:
                return None
            try:
                return await self.tier1.generate_caption(
                    ctx.image_path,  # type: ignore[arg-type]
                    source_caption_hint=ctx.source_caption,
                    image_payload=ctx.image_payload,
                )
            except Exception as e:
                log.warning("caption_task_failed", record_id=ctx.record_id, error=str(e))
                return None

        async def _structure_task() -> StructureOutput | None:
            if not run_structure:
                return None
            from data_forge.data.domain_tagger import _GENERAL_DESIGN_SOURCES
            if rec.domain == "general_design" or rec.source_dataset in _GENERAL_DESIGN_SOURCES:
                return StructureOutput(
                    elements=[],
                    layout_type="freeform",
                    hierarchy_depth=0,
                    background_style="image",
                )
            return await self._extract_structure_with_retry(ctx)

        async def _ocr_task() -> Any | None:
            if not run_ocr or self.ocr is None:
                return None
            try:
                return await self.ocr.extract_text(
                    ctx.image_path,  # type: ignore[arg-type]
                    image_payload=ctx.image_payload,
                )
            except Exception as e:
                log.warning("ocr_task_failed", record_id=ctx.record_id, error=str(e))
                return None

        # Execute VLM queries concurrently so vLLM groups them in continuous batch
        # and achieves prefix cache hits on identical vision tokens.
        caption_res, structure_res, ocr_res = await asyncio.gather(
            _caption_task(),
            _structure_task(),
            _ocr_task(),
        )

        # Validate stage outputs
        if run_recaption and caption_res is None:
            ctx.status = "excluded_failed"
            ctx.failure_reason = "Caption inference failed"
            ctx.exclusion_reason = "inference_failed"
            return ctx

        if run_structure and structure_res is None:
            ctx.status = "excluded_failed"
            ctx.failure_reason = "Structure extraction failed after retries or timeout"
            ctx.exclusion_reason = "structure_extraction_failed"
            return ctx

        if run_ocr and ocr_res is None:
            ctx.status = "excluded_failed"
            ctx.failure_reason = "OCR returned no structured result"
            ctx.exclusion_reason = "ocr_failed"
            return ctx

        ctx.caption_output = caption_res
        ctx.structure_output = structure_res
        ctx.ocr_output = ocr_res
        if run_structure:
            ctx.status = "structured"
        elif run_recaption:
            ctx.status = "recaptioned"
        else:
            ctx.status = "success"
        return ctx

    async def process_subchunk(
        self,
        records: list[ManifestRecord],
        run_recaption: bool = True,
        run_structure: bool = True,
        run_ocr: bool = True,
        concurrency: int | None = None,
    ) -> list[VLMRecordContext]:
        """Process a subchunk of records with bounded concurrency."""
        if concurrency is None:
            concurrency = (
                self.config.get_stage("s05_recaption").get("batch_size")
                or self.config.get_stage("s06_structure").get("batch_size")
                or 128
            )

        contexts = [
            VLMRecordContext(
                record_id=r.id,
                manifest_record=r,
            )
            for r in records
        ]

        sem = asyncio.Semaphore(concurrency)

        async def _bounded_process(ctx: VLMRecordContext) -> VLMRecordContext:
            async with sem:
                return await self.process_record(
                    ctx,
                    run_recaption=run_recaption,
                    run_structure=run_structure,
                    run_ocr=run_ocr,
                )

        tasks = [asyncio.create_task(_bounded_process(ctx)) for ctx in contexts]
        return await asyncio.gather(*tasks)

    @staticmethod
    def to_manifest_updates(
        contexts: list[VLMRecordContext],
        run_recaption: bool = True,
        run_structure: bool = True,
        run_ocr: bool = True,
    ) -> list[dict[str, Any]]:
        """Convert processed contexts into manifest bulk update dictionaries."""
        updates: list[dict[str, Any]] = []
        for ctx in contexts:
            item: dict[str, Any] = {
                "id": ctx.record_id,
            }
            if ctx.status in ("structured", "recaptioned", "success"):
                if run_structure:
                    item["new_status"] = "structured"
                elif run_recaption:
                    item["new_status"] = "recaptioned"

                if run_recaption and ctx.caption_output:
                    item["caption"] = ctx.caption_output.caption
                    item["caption_output"] = ctx.caption_output.model_dump()
                if run_structure and ctx.structure_output:
                    item["structure_output"] = ctx.structure_output.model_dump()
                if run_ocr and ctx.ocr_output:
                    item["ocr_output"] = (
                        ctx.ocr_output.model_dump()
                        if hasattr(ctx.ocr_output, "model_dump")
                        else ctx.ocr_output
                    )
            else:
                item["new_status"] = "excluded_failed"
                item["reason"] = ctx.failure_reason
                item["exclusion_reason"] = ctx.exclusion_reason

            updates.append(item)
        return updates
