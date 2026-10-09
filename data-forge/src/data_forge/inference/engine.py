"""vLLM engine lifecycle manager — subprocess-based model loading/unloading.

Manages the vLLM server as a subprocess. Each model swap tears down the
previous subprocess and starts a new one to prevent CUDA memory fragmentation.

Also manages non-vLLM models (CLIP, VAEs, VQ tokenizers) loaded directly
via torch/transformers.
"""

from __future__ import annotations

import asyncio
import json
import os
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
        self._vllm_log_handle: Any = None
        self._vllm_log_path: str | None = None
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

        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            raise VLLMServerError("vLLM is not installed. Please install the GPU extra.")

        gpu_mem = str(model_spec.gpu_memory_utilization or 0.95)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_spec.model_id,
            "--host", server_cfg.host,
            "--port", str(server_cfg.port),
            "--api-key", server_cfg.api_key,
            "--max-model-len", str(model_spec.max_model_len),
            "--gpu-memory-utilization", gpu_mem,
            "--dtype", model_spec.dtype,
            "--enable-chunked-prefill",  # SOTA: prevents OOM during long multimodal vision prefill
            "--enable-prefix-caching",   # SOTA: caches system prompts & vision prefixes across pipeline stages
            "--max-num-seqs", str(server_cfg.max_num_seqs),
            "--swap-space", str(getattr(server_cfg, "swap_space_gb", 32)),
            "--enforce-eager",
            "--no-enable-log-requests",  # SOTA: removes stdout/disk IO bottleneck
            "--disable-uvicorn-access-log",  # SOTA: eliminates high-concurrency access log spam
            "--uvicorn-log-level", "warning",  # SOTA: suppresses noisy uvicorn INFO polling logs
            "--performance-mode", "throughput",  # SOTA: batch-oriented scheduling for maximum throughput
        ]

        if "vision" in getattr(model_spec, "capabilities", []):
            cmd.extend(["--limit-mm-per-prompt", json.dumps({"image": 1})])

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

        log_dir = config.resolved_paths.get("logs", config.data_root / config.paths.logs)
        log_path = log_dir / f"vllm_{model_key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._vllm_log_handle = log_path.open("ab")
        self._vllm_log_path = str(log_path)
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["MASTER_ADDR"] = "127.0.0.1"
            env["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "300"
            env["CUDA_MODULE_LOADING"] = "LAZY"
            env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            self._vllm_process = subprocess.Popen(
                cmd,
                stdout=self._vllm_log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception:
            self._vllm_log_handle.close()
            self._vllm_log_handle = None
            self._vllm_log_path = None
            raise
        self._current_model = model_key

        # Wait for health check
        base_url = f"http://{server_cfg.host}:{server_cfg.port}"
        try:
            await self._wait_for_health(
                base_url,
                timeout=server_cfg.startup_timeout_seconds,
                interval=server_cfg.health_check_interval_seconds,
            )
        except Exception:
            await self.stop_vllm()
            raise

        self._client = httpx.AsyncClient(
            base_url=f"{base_url}/v1",
            headers={"Authorization": f"Bearer {server_cfg.api_key}"},
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512, keepalive_expiry=120.0),
        )

        log.info("vllm_ready", model=model_key)

    async def stop_vllm(self) -> None:
        """Gracefully shut down the vLLM server subprocess."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._vllm_process is None:
            if self._vllm_log_handle is not None:
                self._vllm_log_handle.close()
                self._vllm_log_handle = None
            return

        log.info("vllm_stopping", model=self._current_model)

        try:
            import psutil
            try:
                parent = psutil.Process(self._vllm_process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                parent.kill()
                psutil.wait_procs(children + [parent], timeout=10)
            except psutil.NoSuchProcess:
                pass
        except Exception as e:
            log.error("vllm_stop_error", error=str(e))

        self._vllm_process = None
        self._current_model = None
        if self._vllm_log_handle is not None:
            self._vllm_log_handle.close()
            self._vllm_log_handle = None

        # Give CUDA time to release memory
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
                    if self._vllm_log_handle is not None:
                        self._vllm_log_handle.flush()
                    log_tail = ""
                    if self._vllm_log_path:
                        with open(self._vllm_log_path, "rb") as process_log:
                            process_log.seek(0, os.SEEK_END)
                            process_log.seek(max(0, process_log.tell() - 4000))
                            log_tail = process_log.read().decode(errors="replace")
                    raise VLLMServerError(
                        f"vLLM process exited with code {self._vllm_process.returncode}. "
                        f"last process log bytes: {log_tail}"
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

        import os
        import torch

        embed_spec = config.models.get("embeddings")
        if not embed_spec:
            raise ValueError("No 'embeddings' model configured in models.yaml")

        device = embed_spec.device if hasattr(embed_spec, "device") else "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for production CLIP inference")
        try:
            from transformers import CLIPModel, CLIPProcessor

            log.info("clip_loading", model_id=embed_spec.model_id, device=device)
            self._clip_processor = CLIPProcessor.from_pretrained(embed_spec.model_id)

            # Use SDPA (Scaled Dot Product Attention) on RTX A6000 Ampere GPU for maximum speed
            attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
            try:
                self._clip_model = CLIPModel.from_pretrained(
                    embed_spec.model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    attn_implementation=attn_impl,
                ).to(device).eval()
            except Exception:
                self._clip_model = CLIPModel.from_pretrained(
                    embed_spec.model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device).eval()

            log.info("clip_loaded", model_id=embed_spec.model_id, device=device, attn=attn_impl)
            return
        except Exception as e:
            log.error("clip_load_failed", error=str(e), model_id=embed_spec.model_id)
            raise RuntimeError(
                f"Failed to load production CLIP model '{embed_spec.model_id}' on {device}: {e}."
            ) from e


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

            try:
                if spec.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for production VAE encoding")
                target_device = spec.device
                dtype = getattr(torch, spec.dtype.replace("float", "float"))
                if target_device == "cpu" and dtype == torch.float16:
                    dtype = torch.float32

                if key == "z_image_vae":
                    # Tongyi-MAI/Z-Image-Turbo publishes its VAE bundled inside
                    # the diffusion pipeline repo under vae/, not as a bare
                    # AutoencoderKL checkpoint at the repo root.
                    try:
                        from diffusers import AutoencoderKL

                        model = AutoencoderKL.from_pretrained(
                            spec.model_id,
                            subfolder="vae",
                            torch_dtype=dtype,
                            revision=spec.revision,
                        ).to(target_device).eval()
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load real VAE {spec.model_id}@{spec.revision}: {e}"
                        ) from e

                # REMOVED: `elif key == "maskgit_vq"`. This loader always
                # raised RuntimeError by design (Open-MAGVIT2 isn't a
                # transformers-native repo), which meant s08_encoding.py's
                # VQ-token branch failed on every ui_first record and
                # `is_encoding_complete()` was never true for that domain.
                # The encoder entry, its only caller, and this loader were
                # removed together — see models.yaml and s08_encoding.py.

                else:
                    log.warning("unknown_encoder", key=key)
                    continue

                # Deterministic VAE channel-count assertion (only when the config
                # actually pins expected_channels — leave unpinned specs alone
                # rather than silently validating against the wrong numbers).
                # z_image_vae only now — qwen_image_vae was removed along with
                # the frozen-model decision (see models.yaml's encoders comment).
                if key == "z_image_vae" and spec.expected_channels is not None:
                    from data_forge.agents.vae_checker import VAEConfigError

                    actual_channels = getattr(model.config, "latent_channels", None)
                    if actual_channels is not None and actual_channels != spec.expected_channels:
                        raise VAEConfigError(
                            f"{key}: VAE channel count mismatch — expected "
                            f"{spec.expected_channels}, got {actual_channels}. Storage "
                            "budget calculations in pipeline.yaml assume the wrong value."
                        )

                self._encoders[key] = model
                log.info("encoder_loaded", key=key)

            except Exception as e:
                log.error(
                    "encoder_load_failed",
                    key=key,
                    model_id=spec.model_id,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Required encoder '{key}' failed to load; refusing to continue "
                    "with missing artifacts."
                ) from e

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
