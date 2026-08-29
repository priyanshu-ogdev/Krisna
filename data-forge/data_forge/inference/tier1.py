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
    SynthesizedConversation,
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
        caption_prompt = self._config.get_prompt("audit_caption")
        structure_prompt = self._config.get_prompt("audit_structure")

        combined_prompt = (
            f"{caption_prompt}\n\n"
            f"## Caption to Verify\n{caption}\n\n"
            f"---\n\n"
            f"{structure_prompt}\n\n"
            f"## Structural JSON to Verify\n{structure_json}"
        )
        result = await self._client.complete(
            prompt=combined_prompt,
            image_path=image_path,
            schema=AuditOutput,
            max_tokens=1024,
        )
        return result if isinstance(result, AuditOutput) else None

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

    async def synthesize_conversation(
        self, seed_record_id: str, seed_critique_text: str
    ) -> SynthesizedConversation | None:
        """Generate one synthetic planner-training conversation, seeded from
        real UICrit human critique text. Text-only completion — no image.
        See stages/s01_6_planner_synthesis.py for how this fits the broader
        planner-data-completeness fix.
        """
        prompt_template = self._config.get_prompt("planner_conversation_synthesis")
        prompt = (
            f"{prompt_template}\n\n"
            f"## Seed critique (record_id: {seed_record_id})\n{seed_critique_text}\n\n"
            f"Remember: seed_critique_record_id in your output must be exactly "
            f"\"{seed_record_id}\"."
        )
        result = await self._client.complete(
            prompt=prompt,
            schema=SynthesizedConversation,
            max_tokens=1536,
            temperature=0.7,  # Higher than other Tier-1 tasks — this is
                               # generative content synthesis, not extraction/
                               # classification, and wants conversational variety.
        )
        return result if isinstance(result, SynthesizedConversation) else None
