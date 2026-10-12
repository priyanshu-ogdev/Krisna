"""Small runtime abstraction for frozen Planner inference.

Backends own model loading and generation; PlannerBackend owns prompts, RAG,
validation, and product semantics.  This keeps llama.cpp, vLLM, and
Transformers replaceable without duplicating the Planner contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RuntimeGeneration:
    text: str
    runtime: str
    model_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class PlannerRuntime(Protocol):
    runtime_name: str

    async def load(self) -> None: ...

    async def unload(self) -> None: ...

    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float,
                 json_schema: dict | None = None) -> RuntimeGeneration: ...
