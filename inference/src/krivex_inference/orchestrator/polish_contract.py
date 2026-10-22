"""Stable request/result contract shared by every Polish operating mode."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PolishRequest:
    handoff_image_ref: str
    prompt: str
    constraints: dict
    locked_regions: list[dict]
    seed: int | None = None
    width: int | None = None
    height: int | None = None

@dataclass(frozen=True)
class PolishProvenance:
    model_id: str
    pipeline: str
    adapter_path: str | None
    inference_steps: int
    seed: int | None
