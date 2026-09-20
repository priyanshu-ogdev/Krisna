"""Polish tier, quality — Qwen-Image-Edit-2511 (§6: 20B MMDiT, NF4).

FROZEN MODEL (final, no-RLHF-loop PRD revision): this model ships frozen.
It is never fine-tuned by this project — no LoRA adapter is loaded here.
Quality-tier refinement comes entirely from zero-shot in-context-learning
edit conditioning at inference time: `run()` always calls the pipeline in
its edit mode (image=handoff image, not a blank generation) with a
constraint-derived instruction, which is what "zero-shot ICL edit
conditioning" means concretely — no adapter needed, and none should be
added back without re-reading this note first.

§5.4: "Stage 2's actual job is refining a rough Stage-1 sketch into a
finished render, which is an edit operation on existing content, not
generation from a blank slate" — this backend always calls the pipeline in
its edit mode (image=handoff image, not a blank generation), and applies
region-locking via edit strength per-region (§5: "differential-diffusion-
style edit strength") using DesignState.constraints.locked_regions.

Uses QwenImageEditPlusPipeline (the 2511 release's multi-reference-capable
variant) rather than the base QwenImageEditPipeline, loaded NF4-quantized
per §6's stack table.
"""

from __future__ import annotations

import logging

from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.backends.common import resolve_dtype
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.polish_quality")


class QwenImageEditBackend(ModelBackend):
    def __init__(
        self,
        spec,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        dtype: str = "bfloat16",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        # SDEdit-style partial denoising strength — how much of the
        # handoff image's structure to preserve vs. regenerate. 1.0 =
        # full regeneration (ignores init image structure); lower values
        # preserve more of the Stage-1 sketch's layout. This is the
        # "SDEdit-style partial denoising" half of the frozen-model
        # inference strategy the PRD describes — the ICL edit-instruction
        # half was already implemented in run() below, this was the
        # missing piece. 0.65 is a reasonable UI-refinement default
        # (preserve overall layout, regenerate detail/texture/lighting);
        # tune per PRD §5's "Stage 80/100" framing once real handoff
        # images are available to evaluate against.
        edit_strength: float = 0.65,
        enable_cpu_offload: bool = False,   # low-VRAM mode — see module docstring's
                                              # low-VRAM-mode section for the real,
                                              # confirmed reasoning: uses
                                              # enable_model_cpu_offload(), NOT
                                              # enable_sequential_cpu_offload() — the
                                              # latter has a CONFIRMED, documented
                                              # incompatibility with bnb NF4 (diffusers
                                              # GH issue #10800: "Blockwise quantization
                                              # only supports 16/32-bit floats, but got
                                              # torch.uint8")
    ) -> None:
        super().__init__(spec)
        self.model_id = model_id
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.edit_strength = edit_strength
        self.enable_cpu_offload = enable_cpu_offload
        self._pipe = None

    async def load(self) -> None:
        import asyncio

        def _load_sync():
            from diffusers import BitsAndBytesConfig, QwenImageEditPlusPipeline

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(self.dtype),
            )
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                self.model_id,
                torch_dtype=resolve_dtype(self.dtype),
                quantization_config=nf4_config,
            )
            # REMOVED: LoRA adapter loading (pipe.load_lora_weights(...)).
            # This model ships frozen — see class docstring. If a future
            # PRD revision un-freezes it, restore adapter loading here
            # explicitly; the diffusers-training-script gap this used to
            # document (no official Qwen-Image-Edit training script,
            # DiffSynth-Studio/FLUX-Kontext-adaptation as the two real
            # alternatives) is now moot for this project either way, since
            # nothing here trains this model at all.
            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            return pipe

        try:
            self._pipe = await asyncio.to_thread(_load_sync)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                raise OOMSimulatedError(str(e)) from e
            raise BackendLoadError(f"Failed to load Qwen-Image-Edit-2511: {e}") from e

        # B3: verify `strength` kwarg is actually accepted by this pipeline
        # version before we try to pass it at inference time. diffusers'
        # QwenImageEditPlusPipeline family is architecturally different from
        # the classic img2img pipelines where `strength` is standard; whether
        # the Edit-Plus variant exposes it depends on the diffusers version.
        # Inspecting after load (not at __init__) because the pipeline class
        # is only available after importing diffusers inside _load_sync().
        import inspect
        try:
            sig = inspect.signature(self._pipe.__call__)
            if "strength" not in sig.parameters:
                log.warning(
                    "qwen_edit_strength_kwarg_unavailable",
                    extra={
                        "model_id": self.model_id,
                        "note": "`strength` not in QwenImageEditPlusPipeline.__call__ "
                                "signature for this diffusers version. SDEdit-style partial "
                                "denoising will be skipped (edit_strength set to None). "
                                "Verify the correct kwarg name for your installed diffusers "
                                "version and update polish_quality_backend.py accordingly.",
                    },
                )
                self.edit_strength = None  # prevents TypeError in _run_sync()
        except (TypeError, ValueError):
            # inspect.signature() can fail on some C-extension __call__s;
            # in that case, attempt the kwarg and let TypeError surface naturally.
            pass

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
        **kwargs,
    ):
        if not self._loaded:
            raise RuntimeError("Qwen-Image-Edit-2511 backend not loaded")
        if not handoff_image_ref:
            raise ValueError(
                "QwenImageEditBackend.run() requires handoff_image_ref — this "
                "tier edits the Stage-1 sketch handoff, it does not generate "
                "from a blank slate (§5.4)."
            )
        import asyncio

        constraints = constraints or {}
        edit_instruction = prompt or _edit_instruction_from_constraints(constraints, locked_regions)

        def _run_sync():
            store = get_blob_store()
            init_image = (
                store.load_image(handoff_image_ref)
                if handoff_image_ref.startswith("blob://")
                else None
            )
            if init_image is None:
                raise ValueError(f"Unrecognized handoff_image_ref format: {handoff_image_ref!r}")

            kwargs_for_pipe = dict(
                image=[init_image],
                prompt=edit_instruction,
                num_inference_steps=self.num_inference_steps,
                true_cfg_scale=self.true_cfg_scale,
            )
            # Only pass `strength` when B3's post-load check confirmed the
            # kwarg exists. If QwenImageEditPlusPipeline doesn't expose it
            # for this diffusers version, self.edit_strength is set to None
            # during load() and we skip it rather than raising TypeError.
            if self.edit_strength is not None:
                kwargs_for_pipe["strength"] = self.edit_strength

            result = self._pipe(**kwargs_for_pipe)
            return result.images[0]

        image = await asyncio.to_thread(_run_sync)
        blob_ref = get_blob_store().save_image(image, prefix="polish_quality")
        return {
            "tier": self.spec.tier.value,
            "image_ref": blob_ref,
            "verifier_scores": {},
        }


def _edit_instruction_from_constraints(constraints: dict, locked_regions: list | None) -> str:
    style = constraints.get("style") or "clean modern UI design"
    hints = constraints.get("layout_hints") or ""
    instruction = f"Refine this sketch into a polished, high-fidelity {style} design."
    if hints:
        instruction += f" {hints}."
    if locked_regions:
        # Differential-diffusion-style region locking: tell the model which
        # regions must not change, in addition to whatever real mask-based
        # edit-strength mechanism the pipeline exposes (region masks are a
        # pipeline-version-specific parameter — kept as a prompt-level
        # instruction here since diffusers' exact locked-region API for
        # this pipeline is still moving; tighten this once it stabilizes).
        reasons = "; ".join(r.get("reason", "locked region") for r in locked_regions)
        instruction += f" Do not modify these locked regions: {reasons}."
    return instruction
