"""Production-grade Planner runtime powered by llama.cpp and GGUF.

Provides zero-copy memory-mapped loading, hardware-calibrated GPU layer offloading,
FlashAttention KV cache optimization, grammar-guided JSON schema decoding,
and hybrid UICrit RAG retrieval.

References & Citations:
- Gerganov, G., et al. (2023). llama.cpp: Port of Facebook's LLaMA model in C/C++.
  https://github.com/ggml-org/llama.cpp (Demonstrating zero-copy mmap and unified GGUF execution).
- Geng, S., Josifoski, M., Peyrard, M., & West, R. (EMNLP 2023).
  Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.
  arXiv:2305.13971.
- Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models.
  arXiv:2307.09702.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
  arXiv:2307.08691.
- Lin, J., et al. (MLSys 2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from krisna_inference.backends.model_loader import (
    LlamaCppConfig,
    build_calibrated_planner_config,
    discover_mmproj_model,
    discover_planner_model,
    probe_cuda_hardware,
)
from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.planner_rag import UICritRAGIndex
from krisna_inference.backends.planner_schema import PlannerResponse
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend

log = logging.getLogger("krisna_inference.backends.planner_backend_llama")

_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_IMAGE_PIXELS = 100_000_000


class PlannerBackendLlama(ModelBackend):
    """High-performance, frozen Planner backend running under llama.cpp."""

    runtime_name = "llama.cpp"

    def __init__(
        self,
        spec,
        *,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        rag_corpus_dir: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        flash_attn: bool | None = None,
    ):
        super().__init__(spec)
        self.model_path = model_path or os.environ.get("KRISNA_PLANNER_GGUF")
        self.mmproj_path = mmproj_path or os.environ.get("KRISNA_PLANNER_MMPROJ")
        self.rag_corpus_dir = rag_corpus_dir
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.flash_attn = flash_attn

        self._llama = None
        self._grammar = None
        self._rag_index = UICritRAGIndex()
        self._load_telemetry: dict[str, Any] = {}

    async def load(self) -> None:
        """Load GGUF weights with zero-copy mmap and hardware calibration."""
        resolved_path = discover_planner_model(self.model_path)
        resolved_mmproj = discover_mmproj_model(self.mmproj_path)

        if not resolved_path:
            # Check if running in synthetic / test mode
            if os.environ.get("KRISNA_USE_REAL_BACKENDS") != "1":
                log.info("No Planner GGUF discovered; running PlannerBackendLlama in synthetic mode.")
                self._loaded = True
                return
            raise BackendLoadError(
                "No verified Planner GGUF artifact found. "
                "Download weights using: python scripts/inference/download_planner_models.py "
                "--repo-id ggml-org/Qwen3.5-9B-GGUF --filename Qwen3.5-9B-Q4_K_M.gguf"
            )

        self.model_path = str(resolved_path)
        if resolved_mmproj:
            self.mmproj_path = str(resolved_mmproj)

        def _load_sync():
            t0 = time.perf_counter()
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                if os.environ.get("KRISNA_USE_REAL_BACKENDS") != "1":
                    log.warning("llama-cpp-python not installed; operating in synthetic mock mode.")
                    return None
                raise BackendLoadError(
                    "Install llama-cpp-python for consumer Planner runtime: "
                    "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
                ) from exc

            hw = probe_cuda_hardware()
            ctx = self.n_ctx or int(os.environ.get("KRISNA_PLANNER_CTX", "8192"))
            gpu_layers = (
                self.n_gpu_layers
                if self.n_gpu_layers is not None
                else int(os.environ.get("KRISNA_PLANNER_GPU_LAYERS", str(hw["recommended_gpu_layers"])))
            )
            flash_attn_val = (
                self.flash_attn
                if self.flash_attn is not None
                else (os.environ.get("KRISNA_PLANNER_FLASH_ATTN", "1") == "1" and hw["supports_flash_attn"])
            )

            config = LlamaCppConfig(
                model_path=str(self.model_path),
                mmproj_path=str(self.mmproj_path) if self.mmproj_path else None,
                n_ctx=ctx,
                n_gpu_layers=gpu_layers,
                use_mmap=self.use_mmap,
                use_mlock=self.use_mlock,
                flash_attn=flash_attn_val,
                n_batch=512,
            )

            kwargs = config.to_llama_kwargs()
            log.info("Initializing llama.cpp with config: %s", {k: v for k, v in kwargs.items() if k != "model_path"})
            llama_instance = Llama(**kwargs)
            load_time = time.perf_counter() - t0

            # Pre-compile JSON schema grammar if LlamaGrammar is available
            grammar_instance = None
            try:
                from llama_cpp import LlamaGrammar
                schema_str = json.dumps(PlannerResponse.model_json_schema())
                grammar_instance = LlamaGrammar.from_json_schema(schema_str)
                log.info("Compiled native GBNF grammar from PlannerResponse schema (Geng et al., 2023).")
            except Exception as e:
                log.debug("LlamaGrammar compilation skipped or unsupported (%s); using response_format json_object.", e)

            return llama_instance, grammar_instance, {"load_time_seconds": round(load_time, 3), "config": kwargs}

        res = await asyncio.to_thread(_load_sync)
        if res is not None:
            self._llama, self._grammar, self._load_telemetry = res

        if self.rag_corpus_dir:
            corpus = Path(self.rag_corpus_dir) / "planner_rag_corpus/uicrit_critiques.jsonl"
            if corpus.exists():
                self._rag_index = await asyncio.to_thread(UICritRAGIndex.from_jsonl, corpus)
        self._loaded = True

    async def unload(self) -> None:
        """Unload llama.cpp model context and release host/VRAM buffers."""
        self._llama = None
        self._grammar = None
        self._rag_index = UICritRAGIndex()
        self._loaded = False
        self._load_telemetry = {}

    def _prompt(
        self,
        message: str,
        constraints: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        critique: Any,
        review: bool,
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Construct constrained system and conversation messages with UICrit RAG."""
        image_payload = self._load_image_payload(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )
        retrieved = self._rag_index.retrieve(message, k=3) if message else []
        system = (
            "You are Krisna's frozen design Planner. Return one valid JSON object conforming to "
            "the krisna.delta.v1 specification with keys: schema_version, reply_text, "
            "design_state_delta, retrieved_critique_ids, and runtime_metadata. "
            "Never invent unsupported tool calls. Active constraints: " + str(constraints or {})
        )
        if review:
            system += " Review the current visual wireframe and layout tree, identify defects, and propose precise updates."
        if critique:
            system += " Prior review: " + str(critique)
        if retrieved:
            system += " Human UICrit design grounding: " + " | ".join(r.note for r in retrieved)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            role = "assistant" if turn.get("role") == "planner" else turn.get("role", "user")
            content = turn.get("content")
            if content:
                messages.append({"role": role, "content": content})

        user_content = message or ("Audit and review the active design state." if review else "Continue the design.")
        if image_payload is None:
            messages.append({"role": "user", "content": user_content})
        else:
            raw_image, mime_type = image_payload
            encoded_image = base64.b64encode(raw_image).decode("ascii")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                            },
                        },
                    ],
                }
            )
        return messages, retrieved

    @staticmethod
    def _image_source(
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        image_bytes: bytes | bytearray | memoryview | None,
    ) -> str | os.PathLike[str] | bytes | bytearray | memoryview | None:
        """Select one image input while rejecting ambiguous caller data."""
        refs = [ref for ref in (image_ref, wireframe_ref) if ref not in (None, "")]
        if image_bytes is not None and refs:
            raise ValueError("Provide image_bytes or an image/wireframe reference, not both")
        if len(refs) > 1 and refs[0] != refs[1]:
            raise ValueError("image_ref and wireframe_ref must identify the same image when both are provided")
        return image_bytes if image_bytes is not None else (refs[0] if refs else None)

    @staticmethod
    def _read_file_bytes(path: Path) -> bytes:
        max_bytes = int(os.environ.get("KRISNA_PLANNER_MAX_IMAGE_BYTES", str(_DEFAULT_MAX_IMAGE_BYTES)))
        if max_bytes <= 0:
            raise ValueError("KRISNA_PLANNER_MAX_IMAGE_BYTES must be positive")
        try:
            with path.open("rb") as handle:
                payload = handle.read(max_bytes + 1)
        except OSError as exc:
            raise ValueError(f"Unable to read Planner image {path}: {exc}") from exc
        if len(payload) > max_bytes:
            raise ValueError(f"Planner image exceeds the {max_bytes}-byte safety limit")
        return payload

    @classmethod
    def _validate_image_bytes(cls, payload: bytes) -> tuple[bytes, str]:
        """Verify image bytes and return them with a MIME type for llama.cpp."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("Pillow is required to attach a Planner review image") from exc

        try:
            with Image.open(BytesIO(payload)) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
                max_pixels = int(
                    os.environ.get("KRISNA_PLANNER_MAX_IMAGE_PIXELS", str(_DEFAULT_MAX_IMAGE_PIXELS))
                )
                if max_pixels <= 0:
                    raise ValueError("KRISNA_PLANNER_MAX_IMAGE_PIXELS must be positive")
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise ValueError("Planner image dimensions exceed the safety limit")
                image.verify()
                mime_type = Image.MIME.get(image_format)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid Planner review image: {exc}") from exc

        if not mime_type:
            raise ValueError(f"Unsupported Planner review image format: {image_format or 'unknown'}")
        return payload, mime_type

    @classmethod
    def _load_image_payload(
        cls,
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        image_bytes: bytes | bytearray | memoryview | None,
    ) -> tuple[bytes, str] | None:
        """Resolve a blob/local reference or byte payload without accepting URLs."""
        source = cls._image_source(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )
        if source is None:
            return None

        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
        else:
            source_text = os.fspath(source)
            if source_text.startswith("blob://"):
                blob_store = get_blob_store()
                root = blob_store.root.resolve()
                candidate = (root / source_text.removeprefix("blob://")).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError("Blob image reference escapes the configured blob store") from exc
                if not candidate.is_file():
                    raise ValueError(f"Planner image blob not found: {source_text}")
                payload = cls._read_file_bytes(candidate)
            else:
                candidate = Path(source_text)
                if not candidate.is_file():
                    raise ValueError(f"Planner local image not found: {candidate}")
                payload = cls._read_file_bytes(candidate)

        return cls._validate_image_bytes(payload)

    def _require_strict_visual_review(self, *, image_ref: Any, wireframe_ref: Any, image_bytes: Any) -> None:
        """Reject a text-only fallback when production review claims visual grounding."""
        source = self._image_source(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )
        if source is None:
            raise ValueError(
                "Planner visual review requires a valid image_ref, wireframe_ref, or image_bytes"
            )
        if not self.mmproj_path or not Path(self.mmproj_path).is_file():
            raise RuntimeError(
                "Planner visual review requires a matching multimodal projector (mmproj)"
            )
        if self._llama is None:
            raise RuntimeError("Planner visual review requires a loaded llama.cpp multimodal runtime")

    async def run(
        self,
        message: str = "",
        constraints: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        prior_critique: Any = None,
        review: bool = False,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute one conversational planning or self-review pass."""
        if not self._loaded:
            raise RuntimeError("Planner backend not loaded")

        if review and os.environ.get("KRISNA_USE_REAL_BACKENDS") == "1":
            self._require_strict_visual_review(
                image_ref=image_ref,
                wireframe_ref=wireframe_ref,
                image_bytes=image_bytes,
            )

        messages, retrieved = self._prompt(
            message,
            constraints or {},
            conversation_history,
            prior_critique,
            review,
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )

        def _generate():
            t0 = time.perf_counter()
            if self._llama is None:
                # Deterministic synthetic response for headless test environments
                synthetic_resp = {
                    "schema_version": "krisna.delta.v1",
                    "reply_text": f"Updated layout design based on: {message or 'review'}",
                    "design_state_delta": {
                        "constraint_updates": {
                            "theme": "modern",
                            "primary_color": "#6366F1",
                            "last_planner_reasoning_note": "Synthetic planner step",
                        }
                    },
                    "retrieved_critique_ids": [r.record_id for r in retrieved],
                    "runtime_metadata": {"runtime": "synthetic_mock"},
                }
                return synthetic_resp, 0.01, 10, 30

            call_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.2 if review else 0.7,
            }
            if self._grammar is not None:
                call_kwargs["grammar"] = self._grammar
            else:
                call_kwargs["response_format"] = {"type": "json_object"}

            result = self._llama.create_chat_completion(**call_kwargs)
            gen_time = time.perf_counter() - t0
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            raw_text = result["choices"][0]["message"]["content"]
            parsed = json.loads(raw_text)
            validated = PlannerResponse.model_validate(parsed)
            return (
                {
                    "schema_version": validated.schema_version,
                    "reply_text": validated.reply_text,
                    "design_state_delta": validated.design_state_delta.model_dump(),
                    "retrieved_critique_ids": [r.record_id for r in retrieved],
                    "runtime_metadata": {
                        "runtime": self.runtime_name,
                        "model_id": self.model_path,
                        "generation_time_seconds": round(gen_time, 3),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "throughput_tok_per_sec": round(output_tokens / max(0.001, gen_time), 1),
                    },
                },
                gen_time,
                input_tokens,
                output_tokens,
            )

        resp, gen_time, in_tok, out_tok = await asyncio.to_thread(_generate)
        return {
            "tier": self.spec.tier.value,
            "reply_text": resp["reply_text"],
            "design_state_delta": resp["design_state_delta"],
            "retrieved_critique_ids": resp.get("retrieved_critique_ids", []),
            "runtime_metadata": resp.get("runtime_metadata", {}),
        }
