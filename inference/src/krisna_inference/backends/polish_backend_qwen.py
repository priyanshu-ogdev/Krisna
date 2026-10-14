"""Qwen-Image-2.1 Polish backend — native pipeline renderer.

The installed QwenImage21Pipeline is authoritative for real inference.  The
legacy Flow-ODE classes below are retained only for clearly labeled synthetic
CPU compatibility output and are never passed to the native pipeline.

IEEE / Top-Tier Research Citations:
1. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (ICLR 2023).
   "Flow Matching for Generative Modeling."
   Simulation-free continuous normalizing flows regressing vector fields on optimal transport paths.
2. Liu, X., Gong, C., & Liu, Q. (ICLR 2023).
   "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow."
   Straight ODE trajectories, few-step generation, and midpoint/Euler integration.
3. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., ... & Rombach, R. (ICML 2024).
   "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis."
   Multimodal DiT velocity formulation, resolution-adjusted time shifting, and flow trajectory dynamics.
4. Brooks, T., Holynski, A., & Efros, A. A. (IEEE/CVF CVPR 2023).
   "InstructPix2Pix: Learning to Follow Image Editing Instructions."
   Dual classifier-free guidance mixing text and visual conditioning vectors:
   v_hat = v(x, 0, 0) + s_I [v(x, c_I, 0) - v(x, 0, 0)] + s_T [v(x, c_I, c_T) - v(x, c_I, 0)].
5. Alibaba Qwen Team (2026).
   "Qwen-Image-2.1: Advanced Visual Foundation Models with Image Conditioning & Flow Matching."
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import numbers
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.common import pick_device, resolve_dtype
from krisna_inference.orchestrator.model_registry import ModelBackend, ModelSpec, Tier

log = logging.getLogger("krisna_inference.backends.polish_backend_qwen")

QWEN_IMAGE_21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE_21_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
QWEN_IMAGE_21_API_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/main/src/diffusers/"
    "pipelines/qwenimage21/pipeline_qwenimage21.py#L509"
)


@dataclass
class FlowODEScheduler:
    """Synthetic-test compatibility scheduler, never the native Qwen scheduler.

    The real Qwen-Image-2.1 pipeline owns its FlowMatchEulerDiscreteScheduler
    and sampling grid.  This helper exists only so CPU/synthetic tests retain
    their deterministic placeholder output; its values must never appear as a
    real execution receipt.

    Grounding:
    - Lipman et al. (ICLR 2023) continuous normalized time interval t in [0, 1].
    - Esser et al. (ICML 2024) resolution time-shift function:
      t_shifted = (shift * t) / (1 + (shift - 1) * t).
    - Brooks et al. (CVPR 2023) dynamic CFG decay to eliminate over-saturation.
    """

    num_steps: int = 28
    strength: float = 0.65
    time_shift: float = 1.0
    guidance_scale: float = 4.0
    image_guidance_scale: float = 1.5
    guidance_schedule: str = "cosine_decay"  # "constant", "linear_decay", "cosine_decay"

    def timesteps(self) -> list[float]:
        """Compute discretized timesteps from t_start to 1.0."""
        t_start = max(0.0, 1.0 - self.strength)
        raw_steps = [t_start + (1.0 - t_start) * (i / max(1, self.num_steps)) for i in range(self.num_steps + 1)]
        if abs(self.time_shift - 1.0) < 1e-5:
            return raw_steps
        # Apply Esser et al. time shift
        s = self.time_shift
        return [(s * t) / (1.0 + (s - 1.0) * t) if t > 0.0 else 0.0 for t in raw_steps]

    def guidance_at(self, t: float) -> float:
        """Compute time-dependent text guidance scale w(t)."""
        if self.guidance_schedule == "constant":
            return self.guidance_scale
        if self.guidance_schedule == "linear_decay":
            # Linear decay from guidance_scale to 1.0 as t -> 1.0
            return 1.0 + (self.guidance_scale - 1.0) * (1.0 - t)
        if self.guidance_schedule == "cosine_decay":
            # Cosine decay preserving crisp high-frequency features early, smooth blending late
            decay = math.cos(0.5 * math.pi * min(1.0, max(0.0, t)))
            return 1.0 + (self.guidance_scale - 1.0) * decay
        return self.guidance_scale


class FlowODEIntegrator:
    """Euler and Midpoint (RK2) numerical ODE integrators for Rectified Flow vector fields.

    Governed by: dx/dt = v_theta(x_t, t, c)
    - Euler step: x_{t+dt} = x_t + dt * v_theta(x_t, t)
    - Midpoint step:
      k1 = v_theta(x_t, t)
      x_mid = x_t + 0.5 * dt * k1
      k2 = v_theta(x_mid, t + 0.5 * dt)
      x_{t+dt} = x_t + dt * k2
    """

    @staticmethod
    def euler_step(x: Any, v: Any, dt: float) -> Any:
        return x + dt * v

    @staticmethod
    def midpoint_step(x: Any, v_fn: Callable[[Any, float], Any], t: float, dt: float) -> Any:
        k1 = v_fn(x, t)
        x_mid = x + 0.5 * dt * k1
        k2 = v_fn(x_mid, t + 0.5 * dt)
        return x + dt * k2


class QwenImage21Backend(ModelBackend):
    """Unified Polish Tier backend powered by Qwen-Image-2.1 with Flow-Matching mechanics."""

    def __init__(
        self,
        spec: ModelSpec,
        model_id: str | None = None,
        lora_adapter_path: str | None = None,
        edit_strength: float = 1.0,
        num_inference_steps: int = 28,
        guidance_scale: float = 1.0,
        image_guidance_scale: float = 1.0,
        guidance_schedule: str = "constant",
        solver_type: str = "native",
        enable_cpu_offload: bool = False,
        dtype: str = "bfloat16",
        device: str | None = None,
        strict: bool = False,
        revision: str = QWEN_IMAGE_21_REVISION,
    ) -> None:
        super().__init__(spec)
        self.strict = strict
        self.model_id = model_id or os.environ.get("KRISNA_POLISH_MODEL_ID", QWEN_IMAGE_21_MODEL_ID)
        self.revision = revision
        if self.strict and self.revision != QWEN_IMAGE_21_REVISION:
            raise ValueError(
                "strict Qwen-Image-2.1 serving requires the pinned model revision "
                f"{QWEN_IMAGE_21_REVISION}"
            )
        # A generic LoRA environment variable is allowed only for the Default
        # tier.  Passing ``None`` for Quality must mean "frozen", not "look up
        # a different global adapter".  The factory still supplies an explicit
        # path when a Default adapter is configured.
        if spec.tier == Tier.POLISH_QUALITY and lora_adapter_path is not None:
            raise ValueError("Qwen Polish Quality is frozen and cannot load a LoRA adapter")
        if lora_adapter_path is not None:
            self.lora_adapter_path = lora_adapter_path
        elif spec.tier == Tier.POLISH_DEFAULT:
            self.lora_adapter_path = os.environ.get("KRISNA_POLISH_LORA_PATH")
        else:
            self.lora_adapter_path = None
        self.edit_strength = edit_strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.guidance_schedule = guidance_schedule
        self.solver_type = solver_type
        self.enable_cpu_offload = enable_cpu_offload
        self.dtype = dtype
        self.device = device or pick_device()
        self._pipeline = None
        self._native_call_parameters: set[str] | None = None
        self._pipeline_class_name: str | None = None
        self._scheduler_class_name: str | None = None

    async def load(self) -> None:
        def _load_sync():
            try:
                import torch
            except ImportError:
                if self.strict:
                    raise RuntimeError(
                        "Qwen-Image-2.1 real mode requires torch and diffusers"
                    )
                log.info("Torch/Diffusers not installed; QwenImage21Backend operating in synthetic test mode.")
                return None

            try:
                import diffusers
            except ImportError:
                if self.strict:
                    raise RuntimeError("Qwen-Image-2.1 real mode requires torch and diffusers")
                log.info("Diffusers not installed; QwenImage21Backend operating in synthetic test mode.")
                return None

            if self.strict and self.device != "cuda":
                raise RuntimeError("strict Qwen-Image-2.1 serving requires device='cuda'")
            if self.strict and not torch.cuda.is_available():
                raise RuntimeError("strict Qwen-Image-2.1 serving requires CUDA")
            if not torch.cuda.is_available() and self.device == "cuda":
                log.info("CUDA unavailable; running QwenImage21Backend in CPU synthetic test mode.")
                return None

            try:
                resolved_dtype = resolve_dtype(self.dtype)
                pipeline_cls = getattr(diffusers, "QwenImage21Pipeline", None)
                if self.strict and pipeline_cls is None:
                    raise RuntimeError("diffusers.QwenImage21Pipeline is required in strict mode")
                if pipeline_cls is None:
                    pipeline_cls = getattr(diffusers, "QwenImagePipeline", None)
                if pipeline_cls is None:
                    pipeline_cls = getattr(diffusers, "AutoPipelineForImage2Image", None)
                if pipeline_cls is None:
                    raise RuntimeError(
                        "No supported Qwen image pipeline class found"
                    )
                log.info("loading_qwen_pipeline", extra={"pipeline_class": pipeline_cls.__name__})
                pipe = pipeline_cls.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    torch_dtype=resolved_dtype,
                )

                # The installed, pinned pipeline is the authority.  Do not
                # infer a call contract from a model name or pass arbitrary
                # kwargs through **kwargs: that can make an integration look
                # successful while silently dropping controls.
                try:
                    parameters = inspect.signature(pipe.__call__).parameters
                except (TypeError, ValueError, AttributeError) as exc:
                    if self.strict:
                        raise RuntimeError("cannot inspect QwenImage21Pipeline.__call__") from exc
                    parameters = {}
                self._native_call_parameters = set(parameters)
                self._pipeline_class_name = type(pipe).__name__
                self._scheduler_class_name = type(getattr(pipe, "scheduler", None)).__name__ \
                    if getattr(pipe, "scheduler", None) is not None else None
                if self.strict and not {"prompt", "image", "num_inference_steps"}.issubset(
                    self._native_call_parameters
                ):
                    raise RuntimeError(
                        "QwenImage21Pipeline.__call__ does not expose the required native parameters"
                    )

                if self.lora_adapter_path:
                    adapter_path = Path(self.lora_adapter_path)
                    if not adapter_path.exists():
                        raise FileNotFoundError(f"Qwen LoRA adapter not found: {adapter_path}")
                    self._validate_and_load_lora(pipe, self.lora_adapter_path)

                if self.enable_cpu_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(self.device)

                return pipe
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"Failed to load the real Qwen-Image-2.1 pipeline: {e}") from e
                log.warning("Failed to load real Qwen-Image-2.1 pipeline (%s); falling back to synthetic mode.", e)
                return None

        self._pipeline = await asyncio.to_thread(_load_sync)
        self._loaded = True

    def _validate_and_load_lora(self, pipe: Any, adapter_path: str) -> None:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Qwen LoRA adapter directory not found: {adapter_dir}")
        manifest_path = adapter_dir / "manifest.json"
        if self.strict and not manifest_path.is_file():
            raise RuntimeError(f"Invalid Qwen LoRA adapter manifest: {manifest_path} is required")
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if not isinstance(manifest, dict):
                    raise TypeError("manifest must be a JSON object")
                if manifest.get("base_model") != self.model_id:
                    raise ValueError(
                        f"adapter base model must be exactly {self.model_id!r}"
                    )
                if self.strict and manifest.get("base_revision") != QWEN_IMAGE_21_REVISION:
                    raise ValueError("strict serving requires the pinned Qwen base revision")
                if not self.strict and manifest.get("base_model") not in (None, self.model_id):
                    raise ValueError(
                        f"adapter base model {manifest.get('base_model')!r} does not match {self.model_id!r}"
                    )
                if manifest.get("conditioning") != "image":
                    raise ValueError("strict serving requires image-conditioned adapter")
                if not self.strict and manifest.get("conditioning") not in (None, "image"):
                    raise ValueError("adapter manifest is not image-conditioned")
                if self.strict and manifest.get("pipeline") != "QwenImage21Pipeline":
                    raise ValueError("strict serving requires QwenImage21Pipeline adapter")
                if not self.strict and manifest.get("pipeline") not in (
                    None,
                    "QwenImage21Pipeline",
                    "QwenImagePipeline",
                ):
                    raise ValueError(f"unsupported Qwen pipeline {manifest.get('pipeline')!r}")
                if self.strict:
                    if manifest.get("schema_version") != 2:
                        raise ValueError("strict serving requires Qwen adapter manifest schema_version=2")
                    if not isinstance(manifest.get("dataset_manifest_sha256"), str) or not manifest.get("dataset_manifest_sha256"):
                        raise ValueError("strict serving requires dataset_manifest_sha256")
                    if not isinstance(manifest.get("split_counts"), dict) or not all(
                        isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        for value in manifest["split_counts"].values()
                    ):
                        raise ValueError("strict serving requires valid split_counts")
                    if not isinstance(manifest.get("lora_target_modules"), list) or not manifest.get("lora_target_modules"):
                        raise ValueError("strict serving requires lora_target_modules")
                log.info("Verified LoRA adapter manifest: %s (run_id=%s)", adapter_path, manifest.get("run_id"))
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"Invalid Qwen LoRA adapter manifest: {manifest_path}") from e
                log.warning("LoRA manifest verification warning: %s", e)

        pipe.load_lora_weights(str(adapter_dir))
        log.info("Loaded Polish LoRA adapter from %s", adapter_path)

    async def unload(self) -> None:
        self._pipeline = None
        self._loaded = False
        def _cleanup():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        await asyncio.to_thread(_cleanup)

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"{self.spec.name} is not loaded")

        handoff_ref = kwargs.get("handoff_image_ref")
        prompt = kwargs.get("prompt") if kwargs.get("prompt") is not None else "High quality professional UI design"
        constraints = kwargs.get("constraints") or {}
        locked_regions = kwargs.get("locked_regions") or []
        is_quality = self.spec.tier == Tier.POLISH_QUALITY or bool(kwargs.get("quality"))
        requested_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        if requested_steps is None:
            requested_steps = self.num_inference_steps
        requested_steps = int(requested_steps)
        if requested_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        steps = 40 if is_quality else requested_steps
        strength = float(kwargs.get("strength", self.edit_strength))
        guidance = float(kwargs.get("guidance_scale", self.guidance_scale))
        true_cfg = float(kwargs.get("true_cfg_scale", guidance))
        seed = kwargs.get("seed")

        if self.strict:
            unsupported = ("strength", "image_guidance_scale", "guidance_schedule", "solver_type")
            supplied_unsupported = [name for name in unsupported if name in kwargs]
            if supplied_unsupported:
                raise ValueError(
                    "native QwenImage21Pipeline does not support: "
                    + ", ".join(supplied_unsupported)
                )
            if self.edit_strength != 1.0 or self.image_guidance_scale != 1.0 or self.guidance_schedule != "constant" or self.solver_type != "native":
                raise ValueError("legacy ODE/SDEdit controls are not supported by native Qwen-Image-2.1")

        if not math.isfinite(strength) or not math.isfinite(guidance) or not math.isfinite(true_cfg):
            raise ValueError("numeric generation controls must be finite")

        scheduler = FlowODEScheduler(
            num_steps=steps,
            strength=strength,
            guidance_scale=guidance,
            image_guidance_scale=self.image_guidance_scale,
            guidance_schedule=self.guidance_schedule,
        )

        blobs = get_blob_store()

        # Load input wireframe / conditioning image
        source_img = None
        if handoff_ref:
            if isinstance(handoff_ref, str) and handoff_ref.startswith("blob://"):
                try:
                    source_img = blobs.load_image(handoff_ref).convert("RGB")
                except (OSError, ValueError, RuntimeError) as e:
                    log.warning("Failed to load handoff blob image %s: %s", handoff_ref, e)
            elif isinstance(handoff_ref, (str, Path)) and Path(handoff_ref).exists():
                from PIL import Image
                with Image.open(handoff_ref) as image:
                    source_img = image.convert("RGB")

        if source_img is None:
            if self.strict:
                raise ValueError("Qwen-Image-2.1 requires a valid image conditioning reference")
            from PIL import Image
            source_img = Image.new("RGB", (1024, 768), color=(245, 247, 250))

        def _generate():
            if self._pipeline is None and self.strict:
                raise RuntimeError("Qwen-Image-2.1 pipeline is unavailable in real mode")
            if self._pipeline is not None:
                try:
                    if self._native_call_parameters is None:
                        try:
                            self._native_call_parameters = set(
                                inspect.signature(self._pipeline.__call__).parameters
                            )
                        except (TypeError, ValueError, AttributeError) as exc:
                            raise RuntimeError("cannot inspect native Qwen pipeline call signature") from exc
                    supported = self._native_call_parameters
                    call = {"prompt": prompt, "image": source_img, "num_inference_steps": steps}
                    if "generator" in supported and seed is not None:
                        import torch
                        generator = torch.Generator(device=self.device).manual_seed(int(seed))
                        call["generator"] = generator
                    elif seed is not None and self.strict:
                        raise ValueError("native Qwen pipeline does not support seeded generation")
                    if "true_cfg_scale" in supported:
                        call["true_cfg_scale"] = true_cfg
                    elif true_cfg != 1.0 and self.strict:
                        raise ValueError("native Qwen pipeline does not support true_cfg_scale")
                    out = self._pipeline(**call)
                    rendered = out.images[0]
                    # Apply region locking if requested
                    if locked_regions:
                        rendered = self._apply_locked_regions(source_img, rendered, locked_regions)
                    return rendered, True, None
                except Exception as e:
                    if self.strict:
                        # Preserve the underlying OOM text so the swap
                        # orchestrator can apply its load/run recovery policy.
                        raise RuntimeError(f"Qwen-Image-2.1 generation failed: {e}") from e
                    log.warning("Pipeline generation failed (%s); using synthetic fallback.", e)
                    fallback_reason = str(e)
                else:
                    fallback_reason = None
            else:
                fallback_reason = "real pipeline unavailable"

            # Synthetic high-quality render fallback for test / CPU environments
            return self._synthesize_render(source_img, prompt, constraints, locked_regions, scheduler), False, fallback_reason

        rendered_image, used_real, fallback_reason = await asyncio.to_thread(_generate)
        output_ref = blobs.save_image(rendered_image, prefix="polish_qwen")

        is_synthetic = not used_real
        live_scheduler = getattr(self._pipeline, "scheduler", None) if used_real else None
        actual_timesteps = self._live_timesteps(live_scheduler)
        actual_steps = len(actual_timesteps) if actual_timesteps is not None else None
        scheduler_class = type(live_scheduler).__name__ if live_scheduler is not None else "FlowODEScheduler"
        receipt = {
            "model": self.model_id,
            "revision": self.revision,
            # Keep the attempted native class visible when a non-strict call
            # failed and fell back; runtime_mode/scheduler_source distinguish
            # that from a successfully executed native pipeline.
            "pipeline_class": self._pipeline_class_name,
            "scheduler_class": scheduler_class,
            "scheduler_source": "live_pipeline" if used_real else "synthetic_compatibility",
            "seed": seed,
            "true_cfg_scale": true_cfg if used_real else None,
            "requested_steps": requested_steps,
            "actual_steps": actual_steps if used_real else steps,
            "actual_timesteps": actual_timesteps if used_real else scheduler.timesteps(),
            "runtime_mode": "real" if used_real else "synthetic",
            "fallback_reason": fallback_reason,
            "native_api_source": QWEN_IMAGE_21_API_SOURCE,
        }
        return {
            "tier": self.spec.tier.value,
            "image_ref": output_ref,
            "prompt_used": prompt,
            "renderer": "qwen_image_2_1",
            "solver": type(live_scheduler).__name__ if used_real and live_scheduler is not None else self.solver_type,
            "steps": steps,
            # Legacy controls are reported only for the synthetic compatibility
            # renderer; native Qwen never receives or applies them.
            "strength": strength if is_synthetic else None,
            "guidance_scale": guidance if is_synthetic else None,
            "guidance_schedule": self.guidance_schedule if is_synthetic else None,
            "timesteps": actual_timesteps if used_real else scheduler.timesteps(),
            "synthetic": is_synthetic,
            "runtime_mode": "synthetic" if is_synthetic else "real",
            "run_receipt": receipt,
            "verifier_scores": {
                # These are deterministic test-mode placeholders only. Real
                # scores are attached by flows.finalize() after VerifierStack
                # evaluation; never present them as model measurements.
                "clip_alignment": 0.92,
                "ocr_readability": 0.94,
                "layout_iou": 0.91,
                "aesthetic": 0.89,
                "handoff_consistency": 0.95,
            } if is_synthetic else {},
        }

    @staticmethod
    def _live_timesteps(scheduler: Any) -> list[Any] | None:
        """Read only timesteps actually installed by a live scheduler."""
        if scheduler is None or not hasattr(scheduler, "timesteps"):
            return None
        values = scheduler.timesteps
        if callable(values):
            return None
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()
        try:
            return list(values)
        except (TypeError, ValueError):
            return None

    def _apply_locked_regions(self, source_img: Any, rendered_img: Any, locked_regions: list[dict]) -> Any:
        """Copy locked source pixels exactly into the rendered image."""
        from PIL import Image

        if source_img.size != rendered_img.size:
            raise ValueError(
                "locked-region exact pixel preservation requires source and rendered images to have identical sizes"
            )
        src = source_img.convert("RGBA")
        dst = rendered_img.convert("RGBA")

        # Create binary mask of locked regions
        mask = Image.new("L", dst.size, color=0)
        w, h = dst.size

        for region in locked_regions:
            if isinstance(region, dict) and "bbox" in region:
                if any(key in region for key in ("x", "y", "w", "h")):
                    raise ValueError("locked region cannot contain both bbox and x/y/w/h")
                bbox = region["bbox"]
            elif isinstance(region, (list, tuple)):
                bbox = region
            elif isinstance(region, dict) and all(key in region for key in ("x", "y", "w", "h")):
                bbox = (region["x"], region["y"], region["w"], region["h"])
            else:
                raise ValueError("locked region must contain exactly one valid bbox representation")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("locked region bbox must contain four numeric values")
            if any(isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) for value in bbox):
                raise ValueError("locked region bbox values must be finite numbers")
            bx, by, bw, bh = (float(value) for value in bbox)
            normalized = all(0.0 <= value <= 1.0 for value in (bx, by, bw, bh))
            if any(value > 1.0 for value in (bx, by, bw, bh)) and any(
                0.0 < value < 1.0 for value in (bx, by, bw, bh)
            ):
                raise ValueError("locked region mixes normalized and absolute coordinates")
            absolute = all(value >= 0.0 for value in (bx, by, bw, bh)) and any(value > 1.0 for value in (bx, by, bw, bh))
            if not normalized and not absolute:
                raise ValueError("locked region mixes normalized and absolute coordinates")
            if bw <= 0 or bh <= 0:
                raise ValueError("locked region width and height must be positive")
            if normalized:
                x0, y0 = round(bx * w), round(by * h)
                x1, y1 = round((bx + bw) * w), round((by + bh) * h)
                if bx + bw > 1.0 or by + bh > 1.0:
                    raise ValueError("normalized locked region extends outside the image")
            else:
                x0, y0 = round(bx), round(by)
                x1, y1 = round(bx + bw), round(by + bh)
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("locked region is outside the image or empty")
            # PIL's rectangle endpoint is inclusive; slicing is half-open and
            # therefore matches the verifier's [x0:x1, y0:y1] contract.
            mask.paste(255, (x0, y0, x1, y1))

        # A binary mask is intentional: feathering would alter pixels inside a
        # protected region and violate the lock contract.
        blended = Image.composite(src, dst, mask)
        return blended.convert("RGB")

    def _synthesize_render(
        self,
        source_img: Any,
        prompt: str,
        constraints: dict,
        locked_regions: list,
        scheduler: FlowODEScheduler,
    ) -> Any:
        """Synthetic render fallback providing realistic visual polishing and region locking."""
        from PIL import ImageDraw, ImageEnhance

        img = source_img.copy().convert("RGB")
        # Visual polish: subtle contrast and sharpness enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.08)
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(1.15)

        draw = ImageDraw.Draw(img)
        # Overlay subtle accent bar indicating style application
        draw.rectangle([0, 0, img.width, 4], fill=(79, 70, 229))

        if locked_regions:
            img = self._apply_locked_regions(source_img, img, locked_regions)

        return img
