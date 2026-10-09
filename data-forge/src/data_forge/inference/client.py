"""OpenAI-compatible async client wrapper for vLLM.

Provides request batching, retry logic, structured output enforcement,
and multimodal (image + text) message construction.
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from data_forge.logging_setup import get_logger

log = get_logger("inference.client")

T = TypeVar("T", bound=BaseModel)

# Maximum concurrent requests to vLLM (tuned for RTX 6000 / A6000 48GB continuous batching)
_DEFAULT_CONCURRENCY = 256
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0


class ImagePayloadCache:
    """Thread-safe LRU cache for encoded multimodal images.

    Prevents repeated disk reads and base64 string allocations across
    sequential or concurrent pipeline stages (e.g. Recaption, Structure, OCR, Audit).
    Validates file mtime_ns and st_size so on-disk file modifications
    automatically invalidate stale entries.
    """

    def __init__(self, max_size: int = 4096) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[int, int, str, str]] = OrderedDict()

    def get_or_encode(self, image_path: Path) -> tuple[str, str]:
        path_str = str(image_path)
        try:
            stat = image_path.stat()
            mtime_ns = stat.st_mtime_ns
            size_bytes = stat.st_size
        except OSError:
            return self._encode_file(image_path)

        with self._lock:
            cached = self._cache.get(path_str)
            if cached is not None:
                c_mtime, c_size, mime, b64_str = cached
                if c_mtime == mtime_ns and c_size == size_bytes:
                    self._cache.move_to_end(path_str)
                    return mime, b64_str

        mime, b64_str = self._encode_file(image_path)

        with self._lock:
            self._cache[path_str] = (mtime_ns, size_bytes, mime, b64_str)
            self._cache.move_to_end(path_str)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return mime, b64_str

    @staticmethod
    def _encode_file(image_path: Path, max_dimension: int = 2048) -> tuple[str, str]:
        suffix = image_path.suffix.lower()
        file_size = 0
        try:
            file_size = image_path.stat().st_size
        except OSError:
            pass

        try:
            from PIL import Image, ImageOps
            import io

            with Image.open(image_path) as raw_img:
                w, h = raw_img.size
                if max(w, h) <= max_dimension and file_size <= 4 * 1024 * 1024:
                    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/webp" if suffix == ".webp" else "image/png"
                    with open(image_path, "rb") as f:
                        return mime, base64.b64encode(f.read()).decode("ascii")

                # Oversized image: downscale to safe max dimension to prevent VLM vision token overflow and OOM
                img = ImageOps.exif_transpose(raw_img)
                ratio = min(max_dimension / w, max_dimension / h)
                new_w = max(1, int(w * ratio))
                new_h = max(1, int(h * ratio))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img.save(buf, format="PNG", optimize=True)
                    mime = "image/png"
                else:
                    img = img.convert("RGB")
                    img.save(buf, format="JPEG", quality=92, optimize=True)
                    mime = "image/jpeg"
                return mime, base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            # Fallback to direct file read if PIL cannot parse
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/webp" if suffix == ".webp" else "image/png"
            with open(image_path, "rb") as f:
                return mime, base64.b64encode(f.read()).decode("ascii")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_GLOBAL_IMAGE_CACHE = ImagePayloadCache(max_size=4096)


def get_global_image_cache() -> ImagePayloadCache:
    return _GLOBAL_IMAGE_CACHE


# Cache model multimodal capability globally across all client instances
_MODEL_MULTIMODAL_SUPPORT: dict[str, bool] = {}
_MM_PROBE_LOCK = asyncio.Lock()

# Cache Pydantic model JSON schemas across requests
_SCHEMA_CACHE: dict[type[BaseModel], dict[str, Any]] = {}


def _get_cached_schema(schema: type[T]) -> dict[str, Any]:
    s = _SCHEMA_CACHE.get(schema)
    if s is None:
        s = schema.model_json_schema()
        _SCHEMA_CACHE[schema] = s
    return s


def _clean_json_str(content: str) -> str:
    content = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()
    elif content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def _repair_json_str(text: str) -> str:
    # Remove C-style block comments (/* ... */)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove single-line JS-style comments (// ...)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # Remove trailing commas before closing braces or brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def _balance_json_brackets(text: str) -> str:
    """Balance unclosed strings, brackets, and braces in truncated JSON text."""
    text = text.strip()
    text = re.sub(r",\s*$", "", text)

    in_string = False
    escape = False
    stack: list[str] = []

    for char in text:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in "{[":
                stack.append("}" if char == "{" else "]")
            elif char in "}]":
                if stack and stack[-1] == char:
                    stack.pop()

    if in_string:
        text += '"'

    text = re.sub(r',\s*$', '', text)
    text = re.sub(r':\s*$', ': null', text)

    while stack:
        text += stack.pop()

    return text


def _parse_json_robustly(content: str) -> Any:
    cleaned = _clean_json_str(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_repair_json_str(cleaned))
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_repair_json_str(_balance_json_brackets(cleaned)))
    except json.JSONDecodeError:
        pass

    start_obj = cleaned.find("{")
    end_obj = cleaned.rfind("}")
    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")

    candidates = []
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(cleaned[start_obj : end_obj + 1])
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append(cleaned[start_arr : end_arr + 1])
    if start_obj != -1:
        candidates.append(_balance_json_brackets(cleaned[start_obj:]))
    if start_arr != -1:
        candidates.append(_balance_json_brackets(cleaned[start_arr:]))

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json_str(cand))
            except json.JSONDecodeError:
                continue

    return json.loads(cleaned)


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
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self._client = http_client
        self._model_id = model_id
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _probe_multimodal_support(self, sample_image: Path | None) -> bool:
        if sample_image is None or not sample_image.is_file():
            return True
        cached = _MODEL_MULTIMODAL_SUPPORT.get(self._model_id)
        if cached is not None:
            return cached

        async with _MM_PROBE_LOCK:
            cached = _MODEL_MULTIMODAL_SUPPORT.get(self._model_id)
            if cached is not None:
                return cached

            try:
                mime, image_data = await asyncio.to_thread(self._encode_image, sample_image)
                probe_body = {
                    "model": self._model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{image_data}"},
                                },
                                {"type": "text", "text": "ping"},
                            ],
                        }
                    ],
                    "max_tokens": 1,
                    "temperature": 0.0,
                }
                resp = await self._client.post("/chat/completions", json=probe_body)
                if resp.status_code == 400 and "not a multimodal model" in resp.text:
                    log.info("model_detected_as_text_only", model=self._model_id)
                    _MODEL_MULTIMODAL_SUPPORT[self._model_id] = False
                    return False
                else:
                    _MODEL_MULTIMODAL_SUPPORT[self._model_id] = True
                    return True
            except Exception as e:
                log.debug("multimodal_probe_failed_defaulting_true", error=str(e))
                return True

    async def complete(
        self,
        prompt: str,
        image_path: Path | None = None,
        schema: type[T] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        image_payload: tuple[str, str] | None = None,
    ) -> dict[str, Any] | T:
        """Send a completion request to vLLM.

        Args:
            prompt: System/user prompt text.
            image_path: Optional path to an image for multimodal input.
            schema: Optional Pydantic model class for structured output enforcement.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            image_payload: Optional pre-encoded (mime, base64) payload for zero-disk-IO reuse.

        Returns:
            Parsed Pydantic model if schema provided, else raw dict.
        """
        # Probe multimodal support once before encoding
        if image_path is not None and self._model_id not in _MODEL_MULTIMODAL_SUPPORT:
            await self._probe_multimodal_support(image_path)

        # Image loading/base64 encoding happens here, concurrently up to the caller's
        # task limit (e.g. 128), so it never steals time from the strict vLLM semaphore.
        # This prevents disk IO latency from starving the GPU's continuous batching.
        messages = await self._build_messages(prompt, image_path, image_payload=image_payload)
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # DISABLE GUIDED DECODING FSM COMPILATION
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
            
        async with self._semaphore:
            response_data = await self._request_with_retry(body)

        content = response_data["choices"][0]["message"]["content"]

        if schema is not None:
            parsed = _parse_json_robustly(content)
            return schema.model_validate(parsed)

        try:
            return _parse_json_robustly(content)
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
        if not items:
            return []

        # Probe model capabilities once before launching parallel workers
        if self._model_id not in _MODEL_MULTIMODAL_SUPPORT:
            for itm in items:
                img_p_str = itm.get("_image_path")
                if img_p_str:
                    img_p = Path(img_p_str)
                    if img_p.is_file():
                        await self._probe_multimodal_support(img_p)
                        break

        started_at = time.monotonic()
        results: list[dict[str, Any] | T | None] = [None] * len(items)
        next_index = 0

        async def _worker() -> None:
            nonlocal next_index
            while next_index < len(items):
                item_index = next_index
                next_index += 1
                item = items[item_index]
                prompt = prompt_template.format_map(item) if item else prompt_template
                image_path = item.get("_image_path")
                if image_path:
                    image_path = Path(image_path)
                results[item_index] = await self._safe_complete(
                    prompt=prompt,
                    image_path=image_path,
                    schema=schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    item_id=item.get("_id", "unknown"),
                )

        await asyncio.gather(
            *(
                _worker()
                for _ in range(min(self._max_concurrent, len(items)))
            )
        )
        elapsed = time.monotonic() - started_at
        failed = sum(result is None for result in results)
        log.info(
            "inference_batch_completed",
            model=self._model_id,
            requests=len(items),
            failed=failed,
            duration_s=round(elapsed, 3),
            requests_per_second=round(len(items) / elapsed, 3) if elapsed else 0.0,
            max_concurrent=self._max_concurrent,
        )
        return results

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

    async def _build_messages(
        self,
        prompt: str,
        image_path: Path | None,
        image_payload: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with optional image."""
        content: list[dict[str, Any]] = []

        is_mm = _MODEL_MULTIMODAL_SUPPORT.get(self._model_id)
        if image_path is not None or image_payload is not None:
            if is_mm is False:
                raise RuntimeError(
                    f"Visual task error: Model '{self._model_id}' is text-only and does not accept vision inputs. "
                    f"A multimodal VLM (e.g. Qwen/Qwen3.5-9B or Qwen/Qwen3.8-27B) is required."
                )
            if image_payload is not None:
                mime, image_data = image_payload
            elif image_path is not None:
                mime, image_data = await asyncio.to_thread(self._encode_image, image_path)
            else:
                mime, image_data = None, None

            if mime and image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{image_data}",
                    },
                })

        content.append({
            "type": "text",
            "text": prompt,
        })

        return [{"role": "user", "content": content}]

    @staticmethod
    def _encode_image(image_path: Path) -> tuple[str, str]:
        """Read and base64-encode image with thread-safe LRU caching."""
        return _GLOBAL_IMAGE_CACHE.get_or_encode(image_path)

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
                if e.response.status_code == 400 and "not a multimodal model" in e.response.text:
                    _MODEL_MULTIMODAL_SUPPORT[self._model_id] = False
                    log.error(
                        "model_is_not_multimodal_vision",
                        model=self._model_id,
                        error=e.response.text,
                    )
                    raise RuntimeError(
                        f"vLLM server rejected request: Model '{self._model_id}' is not a multimodal model. "
                        f"Cannot run visual pipeline stage without a vision-language model (e.g. Qwen/Qwen3.5-9B or Qwen/Qwen3.8-27B)."
                    ) from e

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

            except (httpx.RequestError, httpx.HTTPError) as e:
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
