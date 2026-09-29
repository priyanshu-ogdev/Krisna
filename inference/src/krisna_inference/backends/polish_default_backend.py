"""Polish tier, default — Z-Image-Turbo (§6: fast path, continuous flow-matching).

Requires diffusers built from git main — `ZImagePipeline` isn't in a tagged
PyPI release yet (see huggingface.co/Tongyi-MAI/Z-Image-Turbo). Turbo is
distilled without classifier-free guidance: guidance_scale MUST be 0.0
(non-zero degrades quality per the model card), and 9 inference steps is
the documented sweet spot (~8 actual DiT forwards).
"""

from __future__ import annotations

import logging

from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.common import resolve_dtype
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.polish_default")


class ZImageTurboBackend(ModelBackend):
    def __init__(
        self,
        spec,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        dtype: str = "bfloat16",
        num_inference_steps: int = 9,
        lora_adapter_path: str | None = None,
    ) -> None:
        super().__init__(spec)
        self.model_id = model_id
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.lora_adapter_path = lora_adapter_path
        self._pipe = None

    async def load(self) -> None:
        import asyncio

        def _load_sync():
            from diffusers import ZImagePipeline

            pipe = ZImagePipeline.from_pretrained(
                self.model_id, torch_dtype=resolve_dtype(self.dtype), low_cpu_mem_usage=False
            )
            if self.lora_adapter_path:
                # Standard diffusers LoRA loading — the output of
                # training/polish's Z-Image LoRA wrapper (see that
                # package's docstring: it trains against Tongyi-MAI/Z-Image,
                # the undistilled base, not Z-Image-Turbo directly, per the
                # community-reported finding that Turbo's distillation
                # gradients are unreliable for LoRA/fine-tuning). Turbo and
                # the undistilled model share the same DiT architecture, so
                # the adapter's module names/shapes should match — but
                # loading a De-Turbo-trained adapter onto the Turbo
                # checkpoint here is UNVERIFIED for actual output quality,
                # only for whether it loads without error.
                pipe.load_lora_weights(self.lora_adapter_path)
                log.info("z_image_lora_applied", extra={"adapter": self.lora_adapter_path})
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
        prompt: str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        if not self._loaded:
            raise RuntimeError("Z-Image-Turbo backend not loaded")
        import asyncio

        constraints = constraints or {}
        effective_prompt = prompt or _prompt_from_constraints(constraints)

        def _run_sync():
            import torch

            generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
            image = self._pipe(
                prompt=effective_prompt,
                height=1024,
                width=1024,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0.0,  # required for Turbo — non-zero degrades quality
                generator=generator,
            ).images[0]
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
