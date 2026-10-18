"""Strict local VLM preprocessing for fetched manifest records.

This module intentionally does not use the historical unified coordinator.  A
preprocessing record is committed only when every requested Tier-1 output is a
real, schema-valid response from the live vLLM session.  In particular, source
metadata is never used as a structure substitute and there is no text-only,
CPU, or freeform-output fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from data_forge.config import PipelineConfig
from data_forge.data.schema_validator import SchemaValidator
from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine
from data_forge.inference.ocr import OCREngine
from data_forge.inference.structured_output import (
    CaptionOutput,
    OCROutput,
    QualityOutput,
    SafetyOutput,
    StructureOutput,
)
from data_forge.inference.tier1 import Tier1Engine
from data_forge.manifest import Manifest, ManifestRecord
from data_forge.utils.path_safety import resolve_data_path


class PreprocessorError(RuntimeError):
    """Base error for strict preprocessing failures."""


class NoLiveVLMEngineError(PreprocessorError):
    """Raised when preprocessing is attempted without a live vLLM engine."""


class OutputValidationError(PreprocessorError):
    """Raised when a VLM response is not valid for its output contract."""


@dataclass(frozen=True)
class RecordPreprocessResult:
    """Auditable outcome for one record."""

    record_id: str
    status: str
    reason: str | None = None
    output_sha256: str | None = None


@dataclass(frozen=True)
class PreprocessResult:
    """Summary returned after the manifest transaction and receipt commit."""

    run_id: str
    receipt_path: Path
    processed: int
    excluded: int
    records: tuple[RecordPreprocessResult, ...]


def _model_dump(value: BaseModel) -> dict[str, Any]:
    """Dump a Pydantic result using the project's Pydantic v2 contract."""

    return value.model_dump(mode="json")


def _output_digest(outputs: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(outputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LocalVLMPreprocessor:
    """Run a strict, all-or-nothing VLM bundle for fetched records.

    ``engine`` must be the object yielded by :meth:`ModelEngine.vllm_session`.
    It is deliberately an explicit dependency: constructing a preprocessor
    never starts a model and never falls back to local text/model inference.
    """

    _OUTPUT_TYPES: ClassVar[dict[str, type[BaseModel]]] = {
        "caption": CaptionOutput,
        "structure": StructureOutput,
        "ocr": OCROutput,
        "quality": QualityOutput,
        "safety": SafetyOutput,
    }
    _SCHEMA_NAMES: ClassVar[dict[str, str | None]] = {
        "caption": "caption_output",
        "structure": "structure_output",
        "ocr": "ocr_output",
        "safety": "safety_output",
        # There is no quality_output.json in the repository; QualityOutput is
        # still validated twice through its Pydantic contract below.
        "quality": None,
    }

    def __init__(
        self,
        config: PipelineConfig,
        manifest: Manifest,
        engine: ModelEngine | Any | None = None,
        *,
        tier1_engine: Tier1Engine | Any | None = None,
        ocr_engine: OCREngine | Any | None = None,
        receipt_path: Path | None = None,
        concurrency: int = 16,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.config = config
        self.manifest = manifest
        self.engine = engine
        self._tier1 = tier1_engine
        self._ocr = ocr_engine
        self.receipt_path = receipt_path
        self.concurrency = concurrency
        self._validator = SchemaValidator(config.schemas_dir)

    def _require_live_engine(self) -> Any:
        if self.engine is None:
            raise NoLiveVLMEngineError(
                "Strict VLM preprocessing requires a live ModelEngine from "
                "ModelEngine.vllm_session(config, 'tier1'); no engine was supplied."
            )

        try:
            client = self.engine.vllm_client
        except Exception as exc:
            raise NoLiveVLMEngineError(
                "Strict VLM preprocessing requires a started vLLM client; "
                f"the supplied engine is not live ({type(exc).__name__}: {exc})."
            ) from exc
        if client is None:
            raise NoLiveVLMEngineError("The supplied ModelEngine has no live vLLM client.")

        # ModelEngine.current_model is the model key passed to vllm_session.
        # Test doubles may omit it, so only reject a known, incompatible value.
        current_model = getattr(self.engine, "current_model", None)
        tier1_spec = self.config.models.get("tier1")
        if tier1_spec is None:
            raise NoLiveVLMEngineError("No configured 'tier1' VLM model is available.")
        capabilities = {str(capability).lower() for capability in tier1_spec.capabilities}
        if capabilities and "vision" not in capabilities:
            raise NoLiveVLMEngineError(
                f"Configured tier1 model '{tier1_spec.model_id}' is not a vision-language model."
            )
        if isinstance(current_model, str) and current_model not in {"tier1", tier1_spec.model_id}:
            raise NoLiveVLMEngineError(
                f"The live engine is serving '{current_model}', not configured tier1 model "
                f"'{tier1_spec.model_id}'."
            )
        return client

    def _engines(self) -> tuple[Any, Any]:
        client = self._require_live_engine()
        tier1 = self._tier1 or Tier1Engine(self.engine, self.config)
        if self._ocr is not None:
            ocr = self._ocr
        else:
            # OCR is intentionally sent through the same configured Tier-1
            # VLM session.  OCREngine is usable when its configured model is
            # the same model; otherwise its model label would be misleading,
            # so use the existing client utility directly with OCROutput.
            ocr_spec = self.config.models.get("ocr")
            tier1_spec = self.config.models["tier1"]
            if ocr_spec is not None and ocr_spec.model_id == tier1_spec.model_id:
                ocr = OCREngine(self.engine, self.config)
            else:
                ocr_client = InferenceClient(
                    http_client=client,
                    model_id=tier1_spec.model_id,
                    max_concurrent=self.config.vllm_server.max_num_seqs,
                    extra_body=tier1_spec.chat_template_kwargs,
                )

                class _Tier1OCREngine:
                    async def extract_text(self, image_path: Path, image_payload: tuple[str, str] | None = None) -> OCROutput:
                        result = await ocr_client.complete(
                            prompt=self_outer.config.get_prompt("ocr_extraction"),
                            image_path=image_path,
                            image_payload=image_payload,
                            schema=OCROutput,
                            max_tokens=self_outer.config.get_stage("s05_ocr_enrichment").get("max_ocr_tokens", 4096),
                        )
                        return result  # type: ignore[return-value]

                self_outer = self
                ocr = _Tier1OCREngine()
        return tier1, ocr

    def _validate_output(self, label: str, value: Any) -> dict[str, Any]:
        expected = self._OUTPUT_TYPES[label]
        if not isinstance(value, expected):
            raise OutputValidationError(
                f"{label} inference returned {type(value).__name__}; expected {expected.__name__}."
            )
        try:
            # Re-validation catches a mutated or hand-built test result and
            # makes the all-or-nothing boundary explicit.
            validated = expected.model_validate(_model_dump(value))
            dumped = _model_dump(validated)
        except (ValidationError, TypeError, ValueError) as exc:
            raise OutputValidationError(f"{label} Pydantic validation failed: {exc}") from exc

        schema_name = self._SCHEMA_NAMES[label]
        if schema_name is not None:
            try:
                valid, errors = self._validator.validate(dumped, schema_name)
            except Exception as exc:
                raise OutputValidationError(f"{label} JSON schema validation could not run: {exc}") from exc
            if not valid:
                detail = "; ".join(errors[:3])
                raise OutputValidationError(f"{label} JSON schema validation failed: {detail}")
        return dumped

    async def _process_one(self, record: ManifestRecord, tier1: Any, ocr: Any) -> tuple[RecordPreprocessResult, dict[str, Any] | None]:
        record_id = record.id
        raw_path = record.scrubbed_image_path or record.image_path
        if not raw_path:
            return RecordPreprocessResult(record_id, "excluded_failed", "image_missing: manifest has no image path"), None
        try:
            image_path = resolve_data_path(self.config.data_root, raw_path)
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
        except Exception as exc:
            return RecordPreprocessResult(record_id, "excluded_failed", f"image_missing: {type(exc).__name__}: {exc}"), None

        try:
            # Build one payload for all requests.  Encoding errors are record
            # failures, never a reason to call a text-only fallback.
            image_payload = await asyncio.to_thread(
                InferenceClient._encode_image,
                image_path,
                self.config.local_vlm.max_image_dimension,
            )
            caption, structure, ocr_output, quality, safety = await asyncio.gather(
                tier1.generate_caption(image_path, source_caption_hint=record.source_caption, image_payload=image_payload),
                tier1.extract_structure(image_path, image_payload=image_payload),
                ocr.extract_text(image_path, image_payload=image_payload),
                tier1.score_quality(image_path),
                tier1.classify_safety(image_path),
            )
            outputs = {
                "caption": self._validate_output("caption", caption),
                "structure": self._validate_output("structure", structure),
                "ocr": self._validate_output("ocr", ocr_output),
                "quality": self._validate_output("quality", quality),
                "safety": self._validate_output("safety", safety),
            }
        except Exception as exc:
            reason = f"inference_or_schema_failure: {type(exc).__name__}: {exc}"
            return RecordPreprocessResult(record_id, "excluded_failed", reason), None

        update = {
            "id": record_id,
            "new_status": "structured",
            "reason": "strict local VLM preprocessing complete",
            "caption": outputs["caption"]["caption"],
            "caption_output": outputs["caption"],
            "structure_output": outputs["structure"],
            "ocr_output": outputs["ocr"],
            "quality_output": outputs["quality"],
            "aesthetic_score": outputs["quality"]["aesthetic_score"],
            "safety_output": outputs["safety"],
            "safety_tier": outputs["safety"]["tier"],
        }
        return RecordPreprocessResult(record_id, "structured", output_sha256=_output_digest(outputs)), update

    def _default_receipt_path(self, run_id: str) -> Path:
        audit_dir = self.config.resolved_paths.get("audit_reports") if self.config.resolved_paths else None
        if audit_dir is None:
            audit_dir = self.config.data_root / self.config.paths.audit_reports
        return Path(audit_dir) / f"vlm_preprocess_{run_id}.json"

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)

    async def process(self, records: Iterable[ManifestRecord] | None = None) -> PreprocessResult:
        """Process fetched records and atomically persist updates and receipt."""

        tier1, ocr = self._engines()
        selected = list(records) if records is not None else self.manifest.query_by_status("fetched")
        run_id = uuid.uuid4().hex
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded(record: ManifestRecord) -> tuple[RecordPreprocessResult, dict[str, Any] | None]:
            async with semaphore:
                return await self._process_one(record, tier1, ocr)

        outcomes = await asyncio.gather(*(bounded(record) for record in selected)) if selected else []
        results = [outcome[0] for outcome in outcomes]
        updates = [outcome[1] for outcome in outcomes if outcome[1] is not None]
        failures = [result for result in results if result.status == "excluded_failed"]
        updates.extend(
            {
                "id": result.record_id,
                "new_status": "excluded_failed",
                "reason": result.reason,
                "exclusion_reason": result.reason,
            }
            for result in failures
        )
        if updates:
            # Manifest.bulk_update_records performs one SQLite transaction for
            # successes and exclusions, so no record can be half-committed.
            self.manifest.bulk_update_records(updates, stage="strict_local_vlm_preprocess")

        receipt = {
            "receipt_version": 1,
            "run_id": run_id,
            "stage": "strict_local_vlm_preprocess",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 6),
            "engine": "live_vllm",
            "model_key": "tier1",
            "model_id": self.config.models["tier1"].model_id,
            "fallbacks_allowed": False,
            "validated_outputs": ["caption", "structure", "ocr", "quality", "safety"],
            "validation_contract": "pydantic_and_json_schema",
            "record_count": len(results),
            "processed": len(results) - len(failures),
            "excluded": len(failures),
            "records": [
                {
                    "record_id": result.record_id,
                    "status": result.status,
                    "reason": result.reason,
                    "output_sha256": result.output_sha256,
                }
                for result in results
            ],
        }
        output_path = self.receipt_path or self._default_receipt_path(run_id)
        self._atomic_write_json(output_path, receipt)
        return PreprocessResult(run_id, output_path, len(results) - len(failures), len(failures), tuple(results))

    @classmethod
    async def run_with_vllm(
        cls,
        config: PipelineConfig,
        manifest: Manifest,
        records: Iterable[ManifestRecord] | None = None,
        **kwargs: Any,
    ) -> PreprocessResult:
        """Run using the mandatory live Tier-1 :class:`ModelEngine` session."""

        async with ModelEngine.vllm_session(config, "tier1") as engine:
            return await cls(config, manifest, engine, **kwargs).process(records)


async def preprocess_fetched_records(
    config: PipelineConfig,
    manifest: Manifest,
    records: Iterable[ManifestRecord] | None = None,
    **kwargs: Any,
) -> PreprocessResult:
    """Convenience entry point that always owns a live Tier-1 vLLM session."""

    return await LocalVLMPreprocessor.run_with_vllm(config, manifest, records, **kwargs)


__all__ = [
    "LocalVLMPreprocessor",
    "NoLiveVLMEngineError",
    "OutputValidationError",
    "PreprocessResult",
    "PreprocessorError",
    "RecordPreprocessResult",
    "preprocess_fetched_records",
]
