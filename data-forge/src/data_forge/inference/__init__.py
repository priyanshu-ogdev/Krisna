"""Inference package: model lifecycle, vLLM client, OCR, and structured outputs."""

from __future__ import annotations

from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine, VLLMServerError
from data_forge.inference.ocr import OCREngine
from data_forge.inference.structured_output import (
    CaptionOutput,
    OCROutput,
    QualityOutput,
    SafetyOutput,
    StructureOutput,
    UIElement,
)

__all__ = [
    "ModelEngine",
    "VLLMServerError",
    "InferenceClient",
    "OCREngine",
    "CaptionOutput",
    "OCROutput",
    "QualityOutput",
    "SafetyOutput",
    "StructureOutput",
    "UIElement",
]
