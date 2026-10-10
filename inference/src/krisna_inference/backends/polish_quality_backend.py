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
its edit mode (image=handoff image, not a blank generation), using
DesignState.constraints.locked_regions.

Region locking is CURRENTLY prompt-text-based only ("Do not modify these
locked regions: ..." appended to the edit instruction, see
_edit_instruction_from_constraints below), not the differential-
diffusion-style per-region edit-strength mechanism §5 originally
describes — confirmed this review pass (docs/review/
36_model_review_4_polish_quality.md), not just an open uncertainty:
QwenImageEditPlusPipeline has no `strength`/mask-based region-control
parameter at all in its real `__call__` signature at this project's
currently pinned diffusers commit. See edit_strength's own constructor
docstring for the full finding.

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

try:
    import diffusers

    if not hasattr(diffusers, "QwenImageEditPlusPipeline"):
        try:
            diffusers.QwenImageEditPlusPipeline = diffusers.QwenImageEditPipeline
        except AttributeError:
            pass
except ImportError:
    pass


class QwenImageEditBackend(ModelBackend):
    def __init__(
        self,
        spec,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        dtype: str = "bfloat16",
        # VERIFIED THIS REVIEW PASS (docs/review/36_model_review_4_polish_quality.md):
        # 40 looks like a deviation from QwenImageEditPlusPipeline.__call__'s
        # own bare signature default of 50 — it isn't. That 50 is just a
        # generic Python fallback baked into the pipeline CLASS, not a
        # recommendation specific to this CHECKPOINT. 40 is the real,
        # official Qwen-Image-Edit-2511 model-card setting — confirmed via
        # two independent sources: the official w3ss GGUF model card's own
        # usage snippet (`"num_inference_steps": 40`) and a hosted-API
        # provider's docs explicitly noting "the upstream model card
        # demonstrates 40 steps with true_cfg_scale 4.0". Both values here
        # were already correct; this comment is the citation that was
        # previously missing, not a value change.
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        # SDEdit-style partial denoising strength — how much of the
        # handoff image's structure to preserve vs. regenerate, IF this
        # pipeline version supports it (see load()'s B3 check).
        #
        # DEFINITIVELY CONFIRMED THIS REVIEW PASS (docs/review/36_model_
        # review_4_polish_quality.md), not just "version-uncertain" as
        # the B3 check below originally framed it: checked out the EXACT
        # commit requirements-inference.txt pins
        # (cc8644b447d8f11074d3df06d0ee0e3e7c91bf75) and read
        # QwenImageEditPlusPipeline.__call__'s real signature directly —
        # `strength` does not exist anywhere in it, zero occurrences.
        # This means, as currently pinned, edit_strength NEVER has any
        # effect — B3's runtime check always fires, always sets this to
        # None, always skips it. This is not a bug in the detection logic
        # (which correctly, defensively discovers this and degrades
        # gracefully rather than raising TypeError) — it's a confirmed
        # fact about this pipeline's real current API surface, previously
        # documented here as an open uncertainty rather than a known
        # non-functional state. Qwen-Image-Edit is an instruction-based
        # edit pipeline (closer to InstructPix2Pix/FLUX-Kontext) rather
        # than an SDEdit partial-noise pipeline — "how much changes" is
        # controlled through the edit INSTRUCTION TEXT and
        # `true_cfg_scale`, not a numeric strength parameter, which may
        # be why this pipeline family never grew one. Kept as a
        # constructor parameter (not removed) specifically because the
        # detection is version-aware and forward-compatible: a future
        # diffusers release could add `strength` without any code change
        # needed here, just this pin moving forward. 0.65 is the value
        # that WOULD apply if a future version adds support — a
        # reasonable UI-refinement default (preserve overall layout,
        # regenerate detail/texture/lighting) per PRD §5's "Stage 80/100"
        # framing, not a currently-active setting.
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
        # version before we try to pass it at inference time. CONFIRMED
        # (docs/review/36_model_review_4_polish_quality.md): at the exact
        # diffusers commit this project currently pins
        # (requirements-inference.txt, cc8644b447d8f11074d3df06d0ee0e3e7c91bf75),
        # QwenImageEditPlusPipeline.__call__ has no `strength` parameter
        # at all — this check WILL always fire against that pin, not a
        # hypothetical. Kept as a runtime check rather than a hardcoded
        # skip specifically so a future diffusers version bump that adds
        # `strength` support is picked up automatically with zero code
        # changes here. Inspecting after load (not at __init__) because
        # the pipeline class is only available after importing diffusers
        # inside _load_sync().
        import inspect
        try:
            sig = inspect.signature(self._pipe.__call__)
            if "strength" not in sig.parameters:
                log.warning(
                    "qwen_edit_strength_kwarg_unavailable",
                    extra={
                        "model_id": self.model_id,
                        "note": "`strength` not in QwenImageEditPlusPipeline.__call__ "
                                "for this diffusers version (confirmed absent at this "
                                "project's currently pinned commit). SDEdit-style partial "
                                "denoising is skipped entirely (edit_strength=None); "
                                "'how much changes' is controlled via the edit instruction "
                                "text and true_cfg_scale instead. If a future diffusers "
                                "pin bump adds `strength` support, this check picks it up "
                                "automatically — no code change needed here.",
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
        # regions must not change. UPDATED (docs/review/
        # 36_model_review_4_polish_quality.md): this is the ONLY
        # region-locking mechanism currently in effect, not "in addition
        # to" a mask-based one — confirmed QwenImageEditPlusPipeline has
        # no mask/region-control parameter at all in its real __call__
        # signature at this project's pinned diffusers commit. Prompt-
        # text instruction is the whole mechanism today; revisit if a
        # future diffusers version adds a real mask-based API.
        reasons = "; ".join(r.get("reason", "locked region") for r in locked_regions)
        instruction += f" Do not modify these locked regions: {reasons}."
    return instruction
