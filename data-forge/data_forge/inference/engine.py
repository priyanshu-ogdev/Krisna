"""vLLM engine lifecycle manager — subprocess-based model loading/unloading.

Manages the vLLM server as a subprocess. Each model swap tears down the
previous subprocess and starts a new one to prevent CUDA memory fragmentation.

Also manages non-vLLM models (CLIP, VAEs, VQ tokenizers) loaded directly
via torch/transformers.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger

log = get_logger("inference.engine")


class VLLMServerError(Exception):
    """Raised when the vLLM server fails to start or respond."""


class ModelEngine:
    """Manages GPU model lifecycle.

    Three model families:
    1. vLLM models (Tier-1, Tier-2, OCR) — served as HTTP subprocess
    2. Embedding models (CLIP) — loaded via transformers
    3. Encoder models (VAEs, VQ) — loaded via transformers/custom
    """

    def __init__(self) -> None:
        self._vllm_process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._current_model: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._encoders: dict[str, Any] = {}

    # ── vLLM Subprocess Management ──────────────────────────────────────

    async def start_vllm(self, config: PipelineConfig, model_key: str) -> None:
        """Start vLLM server subprocess for a given model key."""
        if self._current_model == model_key and self._vllm_process:
            log.info("vllm_already_loaded", model=model_key)
            return

        # Teardown any existing server first
        await self.stop_vllm()

        model_spec = config.models.get(model_key)
        if not model_spec:
            raise ValueError(f"Model key '{model_key}' not found in models.yaml")

        server_cfg = config.vllm_server

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_spec.model_id,
            "--host", server_cfg.host,
            "--port", str(server_cfg.port),
            "--api-key", server_cfg.api_key,
            "--max-model-len", str(model_spec.max_model_len),
            "--gpu-memory-utilization", str(model_spec.gpu_memory_utilization),
            "--dtype", model_spec.dtype,
        ]

        if model_spec.quantization:
            cmd.extend(["--quantization", model_spec.quantization])
        if model_spec.trust_remote_code:
            cmd.append("--trust-remote-code")
        if model_spec.revision != "main":
            cmd.extend(["--revision", model_spec.revision])

        log.info(
            "vllm_starting",
            model=model_key,
            model_id=model_spec.model_id,
            quantization=model_spec.quantization,
            max_model_len=model_spec.max_model_len,
        )

        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._current_model = model_key

        # Wait for health check
        base_url = f"http://{server_cfg.host}:{server_cfg.port}"
        await self._wait_for_health(
            base_url,
            timeout=server_cfg.startup_timeout_seconds,
            interval=server_cfg.health_check_interval_seconds,
        )

        self._client = httpx.AsyncClient(
            base_url=f"{base_url}/v1",
            headers={"Authorization": f"Bearer {server_cfg.api_key}"},
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

        log.info("vllm_ready", model=model_key)

    async def stop_vllm(self) -> None:
        """Gracefully shut down the vLLM server subprocess."""
        if self._vllm_process is None:
            return

        log.info("vllm_stopping", model=self._current_model)

        if self._client:
            await self._client.aclose()
            self._client = None

        try:
            self._vllm_process.terminate()
            try:
                self._vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning("vllm_force_kill", model=self._current_model)
                self._vllm_process.kill()
                self._vllm_process.wait(timeout=10)
        except Exception as e:
            log.error("vllm_stop_error", error=str(e))

        self._vllm_process = None
        self._current_model = None

        # Give CUDA time to release memory
        await asyncio.sleep(2)
        log.info("vllm_stopped")

    async def _wait_for_health(
        self, base_url: str, timeout: int, interval: int
    ) -> None:
        """Poll the vLLM health endpoint until ready or timeout."""
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                # Check if process died
                if self._vllm_process and self._vllm_process.poll() is not None:
                    stderr = ""
                    if self._vllm_process.stderr:
                        stderr = self._vllm_process.stderr.read().decode(errors="replace")
                    raise VLLMServerError(
                        f"vLLM process exited with code {self._vllm_process.returncode}. "
                        f"stderr: {stderr[:2000]}"
                    )
                try:
                    resp = await client.get(f"{base_url}/health", timeout=5)
                    if resp.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(interval)

        raise VLLMServerError(
            f"vLLM server did not become healthy within {timeout}s"
        )

    @property
    def vllm_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("vLLM server not started. Call start_vllm() first.")
        return self._client

    @property
    def current_model(self) -> str | None:
        return self._current_model

    # ── CLIP Embedding Model ────────────────────────────────────────────

    def load_clip(self, config: PipelineConfig) -> None:
        """Load CLIP model for image embeddings."""
        if self._clip_model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        embed_spec = config.models.get("embeddings")
        if not embed_spec:
            raise ValueError("No 'embeddings' model configured in models.yaml")

        log.info("clip_loading", model_id=embed_spec.model_id)
        device = embed_spec.device if hasattr(embed_spec, "device") else "cuda"

        self._clip_processor = CLIPProcessor.from_pretrained(embed_spec.model_id)
        self._clip_model = CLIPModel.from_pretrained(
            embed_spec.model_id,
            torch_dtype=torch.float16,
        ).to(device).eval()

        log.info("clip_loaded", model_id=embed_spec.model_id, device=device)

    def unload_clip(self) -> None:
        """Unload CLIP model and free GPU memory."""
        if self._clip_model is None:
            return

        import gc

        import torch

        del self._clip_model
        del self._clip_processor
        self._clip_model = None
        self._clip_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("clip_unloaded")

    @property
    def clip_model(self) -> Any:
        if self._clip_model is None:
            raise RuntimeError("CLIP not loaded. Call load_clip() first.")
        return self._clip_model

    @property
    def clip_processor(self) -> Any:
        if self._clip_processor is None:
            raise RuntimeError("CLIP not loaded. Call load_clip() first.")
        return self._clip_processor

    # ── Encoder Models (VAEs, VQ) ───────────────────────────────────────

    def load_encoders(self, config: PipelineConfig) -> None:
        """Load all tri-path encoder models."""
        import torch

        for key, spec in config.encoders.items():
            if key in self._encoders:
                continue

            log.info("encoder_loading", key=key, model_id=spec.model_id)

            dtype = getattr(torch, spec.dtype.replace("float", "float"))
            if key == "z_image_vae" or key == "qwen_image_vae":
                from diffusers import AutoencoderKL

                model = AutoencoderKL.from_pretrained(
                    spec.model_id,
                    torch_dtype=dtype,
                    revision=spec.revision,
                ).to(spec.device).eval()

            elif key == "maskgit_vq":
                # Open-MAGVIT2 uses a custom loader
                from transformers import AutoModel

                model = AutoModel.from_pretrained(
                    spec.model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    revision=spec.revision,
                ).to(spec.device).eval()

            else:
                log.warning("unknown_encoder", key=key)
                continue

            self._encoders[key] = model
            log.info("encoder_loaded", key=key)

    def unload_encoders(self) -> None:
        """Unload all encoder models."""
        import gc

        import torch

        for key in list(self._encoders.keys()):
            del self._encoders[key]
        self._encoders.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("encoders_unloaded")

    def get_encoder(self, key: str) -> Any:
        if key not in self._encoders:
            raise RuntimeError(f"Encoder '{key}' not loaded. Call load_encoders() first.")
        return self._encoders[key]

    # ── Context Managers (for orchestrator) ─────────────────────────────

    @staticmethod
    @asynccontextmanager
    async def vllm_session(
        config: PipelineConfig, model_key: str
    ) -> AsyncGenerator[ModelEngine, None]:
        """Context manager that starts vLLM, yields the engine, and stops on exit."""
        engine = ModelEngine()
        try:
            await engine.start_vllm(config, model_key)
            yield engine
        finally:
            await engine.stop_vllm()

    @staticmethod
    @asynccontextmanager
    async def clip_session(
        config: PipelineConfig,
    ) -> AsyncGenerator[ModelEngine, None]:
        """Context manager for CLIP embedding model."""
        engine = ModelEngine()
        try:
            engine.load_clip(config)
            yield engine
        finally:
            engine.unload_clip()

    @staticmethod
    @asynccontextmanager
    async def encoder_session(
        config: PipelineConfig,
    ) -> AsyncGenerator[ModelEngine, None]:
        """Context manager for tri-path encoder models."""
        engine = ModelEngine()
        try:
            engine.load_encoders(config)
            yield engine
        finally:
            engine.unload_encoders()
