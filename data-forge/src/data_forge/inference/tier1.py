"""Tier-1 bulk pipeline engine — Qwen3.6-35B-A3B.

Handles: recaptioning, structural extraction, aesthetic scoring,
safety classification (first pass), license text parsing, audit pass.
"""

from __future__ import annotations

from pathlib import Path

from data_forge.config import PipelineConfig
from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine
from data_forge.inference.structured_output import (
    AuditOutput,
    CaptionOutput,
    LicenseOutput,
    QualityOutput,
    SafetyOutput,
    StructureOutput,
)
from data_forge.logging_setup import get_logger

log = get_logger("inference.tier1")


class Tier1Engine:
    """High-level API for Tier-1 model inference tasks."""

    def __init__(self, engine: ModelEngine, config: PipelineConfig) -> None:
        self._engine = engine
        self._config = config
        self._client = InferenceClient(
            http_client=engine.vllm_client,
            model_id=config.models["tier1"].model_id,
        )

    async def score_quality(
        self, image_path: Path
    ) -> QualityOutput | None:
        prompt = self._config.get_prompt("aesthetic_scoring")
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=QualityOutput,
            max_tokens=512,
        )
        return result if isinstance(result, QualityOutput) else None

    async def classify_safety(
        self, image_path: Path
    ) -> SafetyOutput | None:
        prompt = self._config.get_prompt("safety_classification")
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=SafetyOutput,
            max_tokens=512,
        )
        return result if isinstance(result, SafetyOutput) else None

    async def generate_caption(
        self, image_path: Path, source_caption_hint: str | None = None
    ) -> CaptionOutput | None:
        prompt = self._config.get_prompt("recaption")
        # New: fold in the source dataset's own caption/label (e.g.
        # Screen2Words, PD12M/CC12M metadata) as a prior when one exists —
        # previously captured on ingestion but never actually reached this
        # call at all (see manifest.py's `source_caption` column, added
        # this revision). Explicitly a *hint*, not a ground truth to copy
        # verbatim — the VLM is still expected to produce its own dense,
        # structured caption; the hint just gives it a head start and a
        # sanity check against the source dataset's own label.
        if source_caption_hint:
            prompt = (
                f"{prompt}\n\nThe source dataset provides this original "
                f"label/caption for reference (verify and expand on it, "
                f"don't just repeat it verbatim): \"{source_caption_hint}\""
            )
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=CaptionOutput,
            max_tokens=self._config.get_stage("s05_recaption").get(
                "max_caption_tokens", 512
            ),
        )
        return result if isinstance(result, CaptionOutput) else None

    async def extract_structure(
        self, image_path: Path
    ) -> StructureOutput | None:
        prompt = self._config.get_prompt("structural_extraction")
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=StructureOutput,
            max_tokens=self._config.get_stage("s06_structure").get(
                "max_structure_tokens", 2048
            ),
        )
        return result if isinstance(result, StructureOutput) else None

    async def verify_license(
        self, license_text: str, source_url: str
    ) -> LicenseOutput | None:
        prompt_template = self._config.get_prompt("license_verification")
        prompt = (
            f"{prompt_template}\n\n"
            f"## Source URL\n{source_url}\n\n"
            f"## License Page Content\n{license_text}"
        )
        result = await self._client.complete(
            prompt=prompt,
            schema=LicenseOutput,
            max_tokens=1024,
        )
        return result if isinstance(result, LicenseOutput) else None

    async def audit_record(
        self,
        image_path: Path,
        caption: str,
        structure_json: str,
    ) -> AuditOutput | None:
        from data_forge.inference.structured_output import AuditCaptionOutput, AuditStructureOutput

        caption_prompt = self._config.get_prompt("audit_caption")
        caption_full_prompt = f"{caption_prompt}\n\n## Caption to Verify\n{caption}"

        caption_result = await self._client.complete(
            prompt=caption_full_prompt,
            image_path=image_path,
            schema=AuditCaptionOutput,
            max_tokens=1024,
        )

        structure_prompt = self._config.get_prompt("audit_structure")
        structure_full_prompt = f"{structure_prompt}\n\n## Structural JSON to Verify\n{structure_json}"

        structure_result = await self._client.complete(
            prompt=structure_full_prompt,
            image_path=image_path,
            schema=AuditStructureOutput,
            max_tokens=1024,
        )

        if not isinstance(caption_result, AuditCaptionOutput) or not isinstance(structure_result, AuditStructureOutput):
            return None

        return AuditOutput(
            caption_matches_image=caption_result.caption_matches_image,
            structure_matches_image=structure_result.structure_matches_image,
            quality_issues=[],
            safety_issues=[],
            accuracy_issues=caption_result.accuracy_issues,
            hallucination_issues=caption_result.hallucination_issues,
            overall_pass=caption_result.overall_pass and structure_result.overall_pass,
            confidence=min(caption_result.confidence, structure_result.confidence),
            rationale=f"Caption: {caption_result.rationale} Structure: {structure_result.rationale}",
        )

    async def batch_score_quality(
        self, image_paths: list[Path]
    ) -> list[QualityOutput | None]:
        prompt = self._config.get_prompt("aesthetic_scoring")
        items = [{"_image_path": str(p), "_id": str(p)} for p in image_paths]
        results = await self._client.complete_batch(
            items=items,
            prompt_template=prompt,
            schema=QualityOutput,
            max_tokens=512,
        )
        return [r if isinstance(r, QualityOutput) else None for r in results]

    async def batch_classify_safety(
        self, image_paths: list[Path]
    ) -> list[SafetyOutput | None]:
        prompt = self._config.get_prompt("safety_classification")
        items = [{"_image_path": str(p), "_id": str(p)} for p in image_paths]
        results = await self._client.complete_batch(
            items=items,
            prompt_template=prompt,
            schema=SafetyOutput,
            max_tokens=512,
        )
        return [r if isinstance(r, SafetyOutput) else None for r in results]

    # REMOVED: synthesize_conversation(). It generated the planner's
    # synthetic SFT training conversations for the now-removed
    # s01_6_planner_synthesis.py. The planner ships frozen now (RAG over
    # UICrit's real critique text + constrained JSON decoding, no
    # fine-tune) — see the PRD's no-RLHF-loop revision — so nothing in
    # this pipeline generates synthetic conversational content anymore.
