"""Critic Tier engine — Gemma 4 31B Dense (v10 PRD addition).

Generates UICrit-rubric-style critique data at scale, seeding the
`/ui_critique/` store that later bootstraps DPO preference-pair
construction once the product's own generation loop exists (see this
module's docstring note in s10_5_critic_preference.py for why genuine
A/B preference *pairs* are out of scope for data-forge itself — this
engine produces single-candidate scored critiques, the input DPO needs,
not the pairs DPO trains on).

Architecturally independent of Tier-1/Tier-2: different model family
(Gemma vs. Qwen), different job (aesthetic/critique judgment vs. bulk
extraction), loaded via its own vLLM session so it never competes for
VRAM with the bulk pipeline engine.
"""

from __future__ import annotations

from pathlib import Path

from data_forge.config import PipelineConfig
from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine
from data_forge.inference.structured_output import CritiqueOutput
from data_forge.logging_setup import get_logger

log = get_logger("inference.critic")


class CriticEngine:
    """High-level API for Critic Tier (Gemma 4 31B) inference."""

    def __init__(self, engine: ModelEngine, config: PipelineConfig) -> None:
        self._engine = engine
        self._config = config
        self._client = InferenceClient(
            http_client=engine.vllm_client,
            model_id=config.models["critic"].model_id,
            max_concurrent=8,  # Same conservative concurrency as Tier-2 — larger model
        )

    async def critique(
        self,
        image_path: Path,
        caption: str,
        structure_json: str,
    ) -> CritiqueOutput | None:
        """Produce a scored, structured critique for one record."""
        prompt_template = self._config.get_prompt("critique_scoring")
        prompt = (
            f"{prompt_template}\n\n"
            f"## Caption\n{caption}\n\n"
            f"## Structural JSON\n{structure_json}"
        )
        result = await self._client.complete(
            prompt=prompt,
            image_path=image_path,
            schema=CritiqueOutput,
            max_tokens=1024,
        )
        return result if isinstance(result, CritiqueOutput) else None
