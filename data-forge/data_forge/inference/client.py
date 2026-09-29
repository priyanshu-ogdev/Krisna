"""OpenAI-compatible async client wrapper for vLLM.

Provides request batching, retry logic, structured output enforcement,
and multimodal (image + text) message construction.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from data_forge.logging_setup import get_logger

log = get_logger("inference.client")

T = TypeVar("T", bound=BaseModel)

# Maximum concurrent requests to vLLM
_DEFAULT_CONCURRENCY = 16
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0


class InferenceClient:
    """Async client for vLLM's OpenAI-compatible API.

    Features:
    - Structured output enforcement via Pydantic schemas
    - Multimodal image + text input
    - Automatic retry with exponential backoff
    - Concurrency-limited batching
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        model_id: str,
        max_concurrent: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self._client = http_client
        self._model_id = model_id
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def complete(
        self,
        prompt: str,
        image_path: Path | None = None,
        schema: type[T] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> dict[str, Any] | T:
        """Send a completion request to vLLM.

        Args:
            prompt: System/user prompt text.
            image_path: Optional path to an image for multimodal input.
            schema: Optional Pydantic model class for structured output enforcement.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Parsed Pydantic model if schema provided, else raw dict.
        """
        messages = self._build_messages(prompt, image_path)
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if schema is not None:
            body["extra_body"] = {
                "structured_outputs": {
                    "json": schema.model_json_schema(),
                }
            }

        async with self._semaphore:
            response_data = await self._request_with_retry(body)

        content = response_data["choices"][0]["message"]["content"]

        if schema is not None:
            parsed = json.loads(content)
            return schema.model_validate(parsed)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_text": content}

    async def complete_batch(
        self,
        items: list[dict[str, Any]],
        prompt_template: str,
        schema: type[T] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> list[dict[str, Any] | T | None]:
        """Process a batch of items through the model.

        Each item dict is formatted into the prompt_template via str.format_map().

        Args:
            items: List of dicts with template variables.
            prompt_template: Prompt with {placeholders}.
            schema: Optional Pydantic model for structured output.
            max_tokens: Max tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of results (None for failed items).
        """
        tasks = []
        for item in items:
            prompt = prompt_template.format_map(item) if item else prompt_template
            image_path = item.get("_image_path")
            if image_path:
                image_path = Path(image_path)
            tasks.append(
                self._safe_complete(
                    prompt=prompt,
                    image_path=image_path,
                    schema=schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    item_id=item.get("_id", "unknown"),
                )
            )

        results = await asyncio.gather(*tasks)
        return list(results)

    async def _safe_complete(
        self,
        prompt: str,
        image_path: Path | None,
        schema: type[T] | None,
        max_tokens: int,
        temperature: float,
        item_id: str,
    ) -> dict[str, Any] | T | None:
        """Complete with error handling — returns None on failure."""
        try:
            return await self.complete(
                prompt=prompt,
                image_path=image_path,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            log.error(
                "inference_failed",
                item_id=item_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _build_messages(
        self, prompt: str, image_path: Path | None
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with optional image."""
        content: list[dict[str, Any]] = []

        if image_path is not None:
            image_data = self._encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        return [{"role": "user", "content": content}]

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Read and base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    async def _request_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send request with exponential backoff retry on transient errors."""
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=body,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    # Client error — don't retry
                    raise
                last_error = e
                wait = _RETRY_BACKOFF_BASE ** attempt
                log.warning(
                    "inference_retry",
                    attempt=attempt + 1,
                    status=e.response.status_code,
                    wait_s=wait,
                )
                await asyncio.sleep(wait)

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                wait = _RETRY_BACKOFF_BASE ** attempt
                log.warning(
                    "inference_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    wait_s=wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Inference failed after {_MAX_RETRIES} retries: {last_error}"
        )
