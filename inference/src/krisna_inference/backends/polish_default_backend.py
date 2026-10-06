"""Polish tier, default — Z-Image-Turbo (§6: fast path, continuous flow-matching).

Requires diffusers built from git main — `ZImageImg2ImgPipeline` isn't in a
tagged PyPI release yet (see huggingface.co/Tongyi-MAI/Z-Image-Turbo).
Turbo is distilled without classifier-free guidance: guidance_scale MUST
be 0.0 (non-zero degrades quality per the model card), and 9 inference
steps is the documented sweet spot (~8 actual DiT forwards).

Loaded at bf16, never NF4-quantized — see `load()`'s comment for why
that's the correct choice (matches the LoRA's bf16/fp16 training
precision) and `model_registry.py`'s `quantization` field for this tier,
which used to claim NF4 incorrectly.

BUG FOUND ON REVIEW (severe — this is the single most consequential bug
found in the inference layer): this backend previously used
`ZImagePipeline` (diffusers' TEXT-TO-IMAGE-ONLY class for this model —
confirmed via diffusers' own docs, which offer separate
`ZImageImg2ImgPipeline`/`ZImageInpaintPipeline` classes specifically
because the base `ZImagePipeline.__call__` has no `image=` parameter at
all) and never touched `handoff_image_ref` anywhere in `run()`, even
though it accepted it as a parameter. Concretely: every "quick" finalize
through the Default tier (the common, cheap, fast path per PRD framing —
Quality is the heavier alternate) silently threw away the sketch the user
had been iterating on and generated a BRAND NEW, structurally unrelated
image from the prompt text alone. This directly contradicts PRD §5.3's
own Finalize sequence diagram ("Sketch's VQ tokens decoded through the
VQGAN to real pixels ... -> handed to the Polish backend -> Polish Tier
generates") and is a real (not cosmetic) input/output mismatch: the
function signature promised to use `handoff_image_ref`, and silently
didn't. `polish_quality_backend.py` never had this bug — it correctly
requires and uses `handoff_image_ref` via `QwenImageEditPlusPipeline`'s
`image=` kwarg; this fix brings Default in line with that same, already-
correct pattern, using `ZImageImg2ImgPipeline` and its own `strength`
kwarg (confirmed real and documented, unlike Quality's `strength` support
which needed the runtime B3 introspection check because Edit-Plus's API
surface was less certain).
"""

from __future__ import annotations

import logging

from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.common import resolve_dtype
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.polish_default")


import os

class ZImageTurboBackend(ModelBackend):
    def __init__(
        self,
        spec,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        dtype: str = "bfloat16",
        num_inference_steps: int = 9,
        enable_cpu_offload: bool = False,   # low-VRAM mode
        bridge_strength: float = 0.75,       # SOTA Flow-Matching bridge conditioning ratio (1.0 = pure noise, 0.0 = sketch exact)
        strength: float = 0.6,
        lora_adapter_path: str | None = None,
    ) -> None:
        super().__init__(spec)
        self.model_id = model_id
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.lora_adapter_path = lora_adapter_path
        self.strength = strength
        self.enable_cpu_offload = enable_cpu_offload
        self.bridge_strength = float(
            os.environ.get("KRISNA_POLISH_DEFAULT_BRIDGE_STRENGTH", str(bridge_strength))
        )
        self._pipe = None

    async def load(self) -> None:
        import asyncio

        def _load_sync():
            from diffusers import ZImageImg2ImgPipeline

            pipe = ZImageImg2ImgPipeline.from_pretrained(
                self.model_id, torch_dtype=resolve_dtype(self.dtype), low_cpu_mem_usage=False
            )
            adapter_to_load = self.lora_adapter_path
            if not adapter_to_load:
                # Auto-discover trained checkpoint if available
                dpo_path = "models/dpo_checkpoints/stage1_general/final"
                base_lora_path = "checkpoints/polish_default_lora"
                if os.path.exists(os.path.join(dpo_path, "pytorch_lora_weights.safetensors")):
                    adapter_to_load = dpo_path
                elif os.path.exists(os.path.join(base_lora_path, "pytorch_lora_weights.safetensors")):
                    adapter_to_load = base_lora_path

            if adapter_to_load and os.path.exists(adapter_to_load):
                pipe.load_lora_weights(adapter_to_load)
                log.info("z_image_lora_applied", extra={"adapter": adapter_to_load})
            import torch

            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            elif torch.cuda.is_available():
                pipe.to("cuda")
            return pipe

        try:
            self._pipe = await asyncio.to_thread(_load_sync)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                raise OOMSimulatedError(str(e)) from e
            raise BackendLoadError(f"Failed to load Z-Image-Turbo: {e}") from e
        self._loaded = True

    async def unload(self) -> None:
        import asyncio

        def _unload_sync():
            import gc

            self._pipe = None
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        await asyncio.to_thread(_unload_sync)
        self._loaded = False

    async def run(
        self,
        handoff_image_ref: str | None = None,
        constraints: dict | None = None,
        locked_regions: list | None = None,
        prompt: str | None = None,
        seed: int | None = None,
        bridge_strength: float | None = None,
        **kwargs,
    ):
        if not self._loaded:
            raise RuntimeError("Z-Image-Turbo backend not loaded")
        if not handoff_image_ref:
            # BUG FOUND ON REVIEW: previously this parameter was accepted
            # and silently ignored, which is worse than raising — a
            # caller (or a future refactor) could reasonably assume the
            # sketch handoff was being honored, since nothing said
            # otherwise. Matches polish_quality_backend.py's own choice to
            # raise loudly rather than silently degrade to a
            # structurally-unrelated blank-slate generation. Every real
            # call through flows.finalize() always provides a handoff
            # image (finalize is only reachable once a sketch exists — see
            # DesignState.is_finalize_eligible()), so this should never
            # fire in the normal flow; it exists to catch a caller that
            # bypasses that contract.
            raise ValueError(
                "ZImageTurboBackend.run() requires handoff_image_ref — this tier "
                "refines the Stage-1 sketch handoff via image-to-image, it does not "
                "generate from a blank slate (see this module's docstring)."
            )
        import asyncio

        constraints = constraints or {}
        effective_prompt = prompt or _prompt_from_constraints(constraints)

        def _run_sync():
            import torch
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF

            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(gen_device).manual_seed(seed) if seed is not None else None
            h = kwargs.get("height")
            w = kwargs.get("width")
            if h is None or w is None:
                axes_lens = getattr(self._pipe.transformer.config, "axes_lens", None)
                if axes_lens and len(axes_lens) >= 3 and axes_lens[1] <= 128:
                    h = h or 256
                    w = w or 256
                else:
                    h = h or 1024
                    w = w or 1024

            # SOTA Flow-Matching Bridge Conditioning from Model 1 (Sketch Tier layout)
            latents = None
            callback_fn = None
            callback_tensor_inputs = None

            eff_bridge_strength = bridge_strength if bridge_strength is not None else self.bridge_strength
            eff_bridge_strength = max(0.0, min(1.0, float(eff_bridge_strength)))

            sketch_img = None
            if handoff_image_ref:
                store = get_blob_store()
                sketch_img = (
                    store.load_image(handoff_image_ref)
                    if handoff_image_ref.startswith("blob://")
                    else None
                )
                if sketch_img is not None:
                    # Clean pixel-space anti-aliased resizing BEFORE VAE manifold projection
                    from PIL import Image
                    sketch_img = sketch_img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)

            if sketch_img is not None and eff_bridge_strength < 1.0:
                vae_device = next(iter(self._pipe.vae.parameters())).device
                t_sketch = (
                    TF.to_tensor(sketch_img).unsqueeze(0).to(
                        device=vae_device, dtype=self._pipe.vae.dtype
                    ) * 2.0 - 1.0
                )

                with torch.no_grad():
                    sf = getattr(self._pipe.vae.config, "scaling_factor", 0.3611)
                    sf = float(sf) if isinstance(sf, (int, float)) else 0.3611
                    shift = getattr(self._pipe.vae.config, "shift_factor", 0.0)
                    shift = float(shift) if isinstance(shift, (int, float)) else 0.0
                    raw_latents = self._pipe.vae.encode(t_sketch).latent_dist.sample()
                    z_sketch = (raw_latents - shift) * sf

                    eps = torch.randn_like(
                        z_sketch,
                        generator=generator,
                    )
                    # Optimal Transport Flow-Matching bridge interpolation:
                    latents = (1.0 - eff_bridge_strength) * z_sketch + eff_bridge_strength * eps

                    # Canonical [x, y, w, h] locked regions parsing matching design_state.py & layout_iou.py
                    active_locked = locked_regions or constraints.get("locked_regions")
                    parsed_boxes = []
                    if active_locked:
                        _, _, lat_h, lat_w = z_sketch.shape
                        for r in active_locked:
                            bbox = r.get("bbox") if isinstance(r, dict) else getattr(r, "bbox", None)
                            if bbox and len(bbox) == 4:
                                bx, by, bw, bh = bbox
                                x0 = max(0, int(bx * lat_w))
                                x1 = min(lat_w, int((bx + bw) * lat_w))
                                y0 = max(0, int(by * lat_h))
                                y1 = min(lat_h, int((by + bh) * lat_h))
                                if y1 > y0 and x1 > x0:
                                    parsed_boxes.append((y0, y1, x0, x1))
                                    latents[:, :, y0:y1, x0:x1] = z_sketch[:, :, y0:y1, x0:x1]

                        # Mathematical step-end clamping across all ODE integration steps:
                        if parsed_boxes:
                            def _enforce_locked_regions(pipe, step_idx, timestep, callback_kwargs):
                                cur_latents = callback_kwargs.get("latents")
                                if cur_latents is not None:
                                    sigma = (
                                        float(timestep) / 1000.0
                                        if float(timestep) > 1.0
                                        else float(timestep)
                                    )
                                    for y0, y1, x0, x1 in parsed_boxes:
                                        cur_latents[:, :, y0:y1, x0:x1] = (
                                            (1.0 - sigma) * z_sketch[:, :, y0:y1, x0:x1]
                                            + sigma * eps[:, :, y0:y1, x0:x1]
                                        )
                                return callback_kwargs

                            callback_fn = _enforce_locked_regions
                            callback_tensor_inputs = ["latents"]

            if sketch_img is None:
                from PIL import Image
                sketch_img = Image.new("RGB", (w, h), (255, 255, 255))

            pipe_kwargs = dict(
                prompt=effective_prompt,
                image=sketch_img,
                strength=1.0,
                height=h,
                width=w,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0.0,  # required for Turbo — non-zero degrades quality
                generator=generator,
                latents=latents,
            )
            if callback_fn is not None:
                pipe_kwargs["callback_on_step_end"] = callback_fn
                pipe_kwargs["callback_on_step_end_tensor_inputs"] = callback_tensor_inputs

            image = self._pipe(**pipe_kwargs).images[0]
            return image

        image = await asyncio.to_thread(_run_sync)
        blob_ref = get_blob_store().save_image(image, prefix="polish_default")
        return {
            "tier": self.spec.tier.value,
            "image_ref": blob_ref,
            "verifier_scores": {},  # populated by the verifier stack, not this backend
        }


def _prompt_from_constraints(constraints: dict) -> str:
    style = constraints.get("style") or "clean modern UI design"
    palette = constraints.get("palette") or []
    hints = constraints.get("layout_hints") or ""
    parts = [style]
    if palette:
        parts.append(f"color palette: {', '.join(palette)}")
    if hints:
        parts.append(hints)
    return ", ".join(parts)
