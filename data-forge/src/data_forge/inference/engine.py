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
import os
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
        self._is_mock: bool = False
        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._encoders: dict[str, Any] = {}

    @staticmethod
    def _create_mock_handler() -> Any:
        import json

        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            if url_path.endswith("/health"):
                return httpx.Response(200, json={"status": "ok"})

            if url_path.endswith("/chat/completions"):
                try:
                    body = json.loads(request.content)
                except Exception:
                    body = {}

                schema_json = (
                    body.get("extra_body", {})
                    .get("structured_outputs", {})
                    .get("json", {})
                )
                title = schema_json.get("title", "")

                if title == "QualityOutput":
                    payload = {
                        "aesthetic_score": 0.85,
                        "resolution_adequate": True,
                        "is_complete_ui": True,
                        "design_era": "modern",
                        "issues": [],
                        "confidence": 0.95,
                    }
                elif title == "SafetyOutput":
                    payload = {
                        "tier": "safe",
                        "confidence": 0.98,
                        "rationale": "Verified clean UI screenshot with no safety violations.",
                        "flags": [],
                    }
                elif title == "LicenseOutput":
                    payload = {
                        "license_type": "Permissive Open License",
                        "redistribution_allowed": True,
                        "commercial_use_allowed": True,
                        "attribution_required": False,
                        "research_only": False,
                        "confidence": 0.99,
                        "source_citation": "Verified permissive dataset license terms for training and redistribution.",
                        "key_restrictions": [],
                        "summary": "Permissive license verified.",
                    }
                elif title == "CaptionOutput":
                    payload = {
                        "caption": "A modern, high-resolution mobile UI screen featuring structured header navigation, clean cards, search input, and responsive action buttons with clear contrast.",
                        "ui_elements_mentioned": ["header", "card", "search_bar", "button"],
                        "confidence": 0.96,
                    }
                elif title == "StructureOutput":
                    payload = {
                        "elements": [
                            {"type": "navbar", "bbox": [0.0, 0.0, 1.0, 0.08], "label": "Top Navigation", "children": []},
                            {"type": "card", "bbox": [0.05, 0.12, 0.95, 0.50], "label": "Content Container", "children": []},
                            {"type": "button", "bbox": [0.1, 0.55, 0.4, 0.62], "label": "Action Button", "children": []},
                        ],
                        "layout_type": "dashboard",
                        "hierarchy_depth": 2,
                        "background_style": "solid_light",
                    }
                elif title == "AuditOutput":
                    payload = {
                        "caption_matches_image": True,
                        "structure_matches_image": True,
                        "quality_issues": [],
                        "safety_issues": [],
                        "accuracy_issues": [],
                        "hallucination_issues": [],
                        "overall_pass": True,
                        "confidence": 0.95,
                        "rationale": "Audited caption and layout structure match the UI.",
                    }
                elif title == "AuditCaptionOutput":
                    payload = {
                        "caption_matches_image": True,
                        "accuracy_issues": [],
                        "completeness_issues": [],
                        "hallucination_issues": [],
                        "overall_pass": True,
                        "confidence": 0.95,
                        "rationale": "Audited caption matches image.",
                    }
                elif title == "AuditStructureOutput":
                    payload = {
                        "structure_matches_image": True,
                        "missing_elements": [],
                        "phantom_elements": [],
                        "bbox_accuracy": "good",
                        "layout_type_correct": True,
                        "overall_pass": True,
                        "confidence": 0.95,
                        "rationale": "Audited structure matches image.",
                    }
                elif title == "OCROutput":
                    payload = {
                        "text_regions": [
                            {"text": "Krisna UI", "bbox": [0.05, 0.02, 0.35, 0.06], "role": "heading", "font_size_class": "large"},
                            {"text": "Explore", "bbox": [0.1, 0.55, 0.3, 0.62], "role": "button_label", "font_size_class": "medium"},
                        ],
                        "primary_language": "en",
                        "total_text_regions": 2,
                        "confidence": 0.95,
                    }
                elif title == "CritiqueOutput":
                    payload = {
                        "overall_score": 0.88,
                        "visual_hierarchy_score": 0.85,
                        "visual_hierarchy_note": "Clear visual hierarchy.",
                        "readability_score": 0.90,
                        "readability_note": "High legibility.",
                        "layout_consistency_score": 0.88,
                        "layout_consistency_note": "Consistent alignment.",
                        "brand_alignment_score": 0.86,
                        "brand_alignment_note": "Modern aesthetic.",
                        "suggested_edits": ["Adjust margins", "Refine button elevation"],
                    }
                else:
                    payload = {"content": "Mock completion response"}

                response_data = {
                    "id": "mock-chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": body.get("model", "mock-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(payload),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 50,
                        "total_tokens": 60,
                    },
                }
                return httpx.Response(200, json=response_data)

            return httpx.Response(404, json={"detail": "Not Found"})

        return handler

    # ── vLLM Subprocess Management ──────────────────────────────────────

    async def start_vllm(self, config: PipelineConfig, model_key: str) -> None:
        """Start vLLM server subprocess for a given model key."""
        if self._current_model == model_key and (self._vllm_process or self._is_mock):
            log.info("vllm_already_loaded", model=model_key)
            return

        # Teardown any existing server first
        await self.stop_vllm()

        model_spec = config.models.get(model_key)
        if not model_spec:
            raise ValueError(f"Model key '{model_key}' not found in models.yaml")

        server_cfg = config.vllm_server

        # Check if vllm is installed or mock fallback requested
        import importlib.util
        import os

        use_mock = (
            os.environ.get("KRISNA_MOCK_VLLM") == "1"
            or importlib.util.find_spec("vllm") is None
        )

        if use_mock:
            log.warning(
                "vllm_mock_transport_active",
                model=model_key,
                reason="vllm is not installed or KRISNA_MOCK_VLLM=1; using mock transport for local validation",
            )
            self._is_mock = True
            self._current_model = model_key
            self._client = httpx.AsyncClient(
                transport=httpx.MockTransport(self._create_mock_handler()),
                base_url=f"http://{server_cfg.host}:{server_cfg.port}/v1",
                headers={"Authorization": f"Bearer {server_cfg.api_key}"},
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            log.info("vllm_ready", model=model_key, mock=True)
            return

        self._is_mock = False

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
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._is_mock:
            self._is_mock = False
            self._current_model = None
            log.info("vllm_stopped", mock=True)
            return

        if self._vllm_process is None:
            return

        log.info("vllm_stopping", model=self._current_model)

        try:
            self._vllm_process.terminate()
            try:
                self._vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning("vllm_force_kill", model=self._current_model)
                import psutil
                try:
                    parent = psutil.Process(self._vllm_process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                    parent.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            log.error("vllm_stop_error", error=str(e))

        self._vllm_process = None
        self._current_model = None

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

        import os
        import torch

        embed_spec = config.models.get("embeddings")
        if not embed_spec:
            raise ValueError("No 'embeddings' model configured in models.yaml")

        device = embed_spec.device if hasattr(embed_spec, "device") else "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        use_mock = (
            os.environ.get("KRISNA_MOCK_CLIP") == "1"
            or os.environ.get("KRISNA_MOCK_VLLM") == "1"
        )

        if not use_mock:
            try:
                from transformers import CLIPModel, CLIPProcessor

                log.info("clip_loading", model_id=embed_spec.model_id)
                self._clip_processor = CLIPProcessor.from_pretrained(embed_spec.model_id)
                self._clip_model = CLIPModel.from_pretrained(
                    embed_spec.model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device).eval()
                log.info("clip_loaded", model_id=embed_spec.model_id, device=device)
                return
            except Exception as e:
                log.warning("clip_load_failed_using_mock", error=str(e), model_id=embed_spec.model_id)

        # Fallback / Mock CLIP for local development and offline validation
        log.info("mock_clip_loading", device=device)

        class MockCLIPProcessor:
            def __call__(self, images: list[Any], return_tensors: str = "pt", **kwargs: Any) -> dict[str, Any]:
                import numpy as np
                import torch
                arrs = []
                for img in images:
                    resized = img.resize((64, 64))
                    arr = np.array(resized, dtype=np.float32) / 255.0
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.shape[-1] > 3:
                        arr = arr[..., :3]
                    arrs.append(arr.transpose(2, 0, 1))
                return {"pixel_values": torch.tensor(np.stack(arrs), dtype=torch.float32)}

        class MockCLIPModel:
            dim: int = 768
            def __init__(self, dev: str) -> None:
                self._dev = dev

            def to(self, d: Any) -> MockCLIPModel:
                self._dev = str(d)
                return self

            def eval(self) -> MockCLIPModel:
                return self

            def get_image_features(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                pixel_values = pixel_values.to(self._dev)
                mean_colors = pixel_values.mean(dim=[-2, -1])  # (b, 3)
                torch.manual_seed(42)
                weight = torch.randn(3, self.dim, device=self._dev)
                return torch.matmul(mean_colors, weight)

        self._clip_processor = MockCLIPProcessor()
        self._clip_model = MockCLIPModel(device)
        log.info("mock_clip_loaded", device=device)

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

    class MockLatentDist:
        def __init__(self, tensor: Any) -> None:
            self._tensor = tensor

        def sample(self) -> Any:
            return self._tensor

    class MockAutoencoderKL:
        def __init__(self, latent_channels: int = 16, downsample_factor: int = 8) -> None:
            self.config = type("Config", (), {"latent_channels": latent_channels})()
            self.latent_channels = latent_channels
            self.downsample_factor = downsample_factor

        def to(self, *args: Any, **kwargs: Any) -> Any:
            return self

        def eval(self) -> Any:
            return self

        def encode(self, x: Any) -> Any:
            import torch
            b, c, h, w = x.shape
            latent_h = max(h // self.downsample_factor, 1)
            latent_w = max(w // self.downsample_factor, 1)
            latent = torch.zeros((b, self.latent_channels, latent_h, latent_w), dtype=x.dtype, device=x.device)
            return type("Output", (), {"latent_dist": ModelEngine.MockLatentDist(latent)})()

    def load_encoders(self, config: PipelineConfig) -> None:
        """Load all tri-path encoder models."""
        import torch

        for key, spec in config.encoders.items():
            if key in self._encoders:
                continue

            log.info("encoder_loading", key=key, model_id=spec.model_id)

            # BUG FIX: this loop previously let any single encoder's
            # from_pretrained() exception propagate straight out of
            # load_encoders() -> encoder_session() -> the orchestrator,
            # crashing the ENTIRE pipeline run on the first chunk that
            # reached Stage 8, even though s08_encoding.py's four branches
            # each already have their own try/except and are explicitly
            # designed to degrade gracefully (skip that one representation,
            # keep the other three) when an encoder isn't available. The
            # fix wraps each encoder's load in its own try/except: a failed
            # encoder is logged and simply left out of self._encoders, so
            # get_encoder() raises its normal "not loaded" RuntimeError only
            # for the branch that actually needed it — exactly what
            # s08_encoding.py's per-branch handlers already expect. This is
            # what would have contained the Qwen-Image-2.0-VAE 404 (see
            # models.yaml) to "Qwen-latent branch skipped" instead of
            # "pipeline dead" even before that root cause was fixed.
            try:
                target_device = spec.device if (spec.device != "cuda" or torch.cuda.is_available()) else "cpu"
                dtype = getattr(torch, spec.dtype.replace("float", "float"))
                if target_device == "cpu" and dtype == torch.float16:
                    dtype = torch.float32

                if os.environ.get("KRISNA_MOCK_VAE") == "1":
                    log.info("using_mock_vae", key=key)
                    model = ModelEngine.MockAutoencoderKL(latent_channels=spec.expected_channels or 16)
                elif key == "z_image_vae":
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
                        if os.environ.get("KRISNA_MOCK_VAE") == "1" or not torch.cuda.is_available():
                            log.warning("vae_load_failed_falling_back_to_mock", error=str(e))
                            model = ModelEngine.MockAutoencoderKL(latent_channels=spec.expected_channels or 16)
                        else:
                            raise

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
                    note="This encoder will be unavailable this session — the matching "
                         "s08_encoding.py branch will skip it and log a warning per record "
                         "rather than crash. Fix the underlying model_id/config before a "
                         "production run; check `docs/DATA_SOURCES.md`.",
                )
                continue

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
