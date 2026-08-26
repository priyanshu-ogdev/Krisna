"""Dedicated OCR engine — DeepSeek-OCR-2.

Specialized text-in-image extraction, separate from the general reasoning model.
2026 benchmarks show dedicated OCR models outperform general VLMs on layout/OCR
precision — this is why we don't reuse Tier-1 for OCR.
"""

from __future__ import annotations

from pathlib import Path

from data_forge.config import PipelineConfig
from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine
from data_forge.inference.structured_output import OCROutput
from data_forge.logging_setup import get_logger

log = get_logger("inference.ocr")


class OCREngine:
    """High-level API for dedicated OCR extraction."""

    def __init__(self, engine: ModelEngine, config: PipelineConfig) -> None:
        self._engine = engine
        self._config = config
        self._client = InferenceClient(
            http_client=engine.vllm_client,
            model_id=config.models["ocr"].model_id,
            max_concurrent=16,
        )

    async def extract_text(self, image_path: Path) -> OCROutput | None:
        """Extract all visible text from a UI screenshot."""
        prompt = self._config.get_prompt("ocr_extraction")
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=OCROutput,
            max_tokens=4096,  # OCR can produce long output for text-heavy UIs
        )
        return result if isinstance(result, OCROutput) else None

    async def batch_extract_text(
        self, image_paths: list[Path]
    ) -> list[OCROutput | None]:
        """Batch OCR extraction."""
        prompt = self._config.get_prompt("ocr_extraction")
        items = [{"_image_path": str(p), "_id": str(p)} for p in image_paths]
        results = await self._client.complete_batch(
            items=items,
            prompt_template=prompt,
            schema=OCROutput,
            max_tokens=4096,
        )
        return [r if isinstance(r, OCROutput) else None for r in results]
