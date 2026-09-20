"""Sketch tier backend — loads a MaskGITSketchModel checkpoint.

There is no pretrained checkpoint for this tier (see maskgit_model.py's
docstring — it's the project's single biggest open research risk per PRD
§6.1/§11.1). This backend's load() fails with a clear, actionable error if
no checkpoint path is configured or the file doesn't exist, rather than
silently falling back to random weights, which would make "sketch this"
appear to work while producing meaningless output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.sketch")


class SketchBackend(ModelBackend):
    def __init__(
        self,
        spec,
        checkpoint_path: str | None = None,
        num_rounds: int = 8,
        guidance_scale: float = 3.0,
    ) -> None:
        super().__init__(spec)
        self.checkpoint_path = checkpoint_path
        self.num_rounds = num_rounds
        # Moderate, literature-anchored starting point (no trained
        # checkpoint exists yet to tune this against — see
        # maskgit_model.py's module docstring). 0.0 (no guidance, the
        # pre-Phase-17 behavior) remains available by passing 0.0.
        self.guidance_scale = guidance_scale
        self._model = None
        self._text_embedder = None

    async def load(self) -> None:
        if not self.checkpoint_path:
            raise BackendLoadError(
                "Sketch tier has no checkpoint_path configured. This tier is "
                "trained from scratch (PRD §6, §6.1) — there is no public "
                "checkpoint to download. Train one and pass its path via "
                "SketchBackend(checkpoint_path=...) before this tier can load."
            )
        if not Path(self.checkpoint_path).exists():
            raise BackendLoadError(f"Sketch checkpoint not found: {self.checkpoint_path}")

        import asyncio

        def _load_sync():
            from krisna_inference.backends.common import pick_device
            from krisna_inference.backends.maskgit_model import MaskGITSketchModel

            return MaskGITSketchModel.from_checkpoint(self.checkpoint_path, device=pick_device())

        try:
            self._model = await asyncio.to_thread(_load_sync)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                raise OOMSimulatedError(str(e)) from e
            raise BackendLoadError(f"Failed to load sketch checkpoint: {e}") from e

        # Real prompt conditioning: loads the SAME CLIP embedder singleton
        # training's collate_fn uses (krisna_inference.verifiers.common.
        # get_clip_embedder), guaranteeing the embedding space matches by
        # construction. Falls back to unconditional generation (logged) if
        # CLIP/transformers deps aren't installed, rather than hard-failing
        # the whole backend load — see docs/review/17_sketch_inference_conditioning_and_cfg.md.
        def _load_embedder_sync():
            from krisna_inference.verifiers.common import get_clip_embedder

            embedder = get_clip_embedder()
            embedder.load()
            return embedder

        try:
            self._text_embedder = await asyncio.to_thread(_load_embedder_sync)
        except ImportError:
            log.warning(
                "sketch_text_embedder_unavailable",
                extra={"note": "CLIP/transformers not installed — Sketch tier will generate unconditionally (zero embedding) regardless of prompt text"},
            )
            self._text_embedder = None

        self._loaded = True

    async def unload(self) -> None:
        import asyncio

        def _unload_sync():
            import gc

            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        await asyncio.to_thread(_unload_sync)
        self._loaded = False

    async def run(self, planner_output: dict | None = None, prior_tokens_ref: str | None = None, **kwargs):
        if not self._loaded:
            raise RuntimeError("Sketch backend not loaded")
        import asyncio

        from krisna_inference.backends.blob_store_singleton import get_blob_store

        blobs = get_blob_store()
        prior_tokens = blobs.load_tokens(prior_tokens_ref) if prior_tokens_ref else None

        # The user's actual message, forwarded via **kwargs from
        # SwapOrchestrator.run_conversational_turn (the same kwargs dict
        # PlannerBackend.run(message=...) receives). Grounded with active
        # constraints (style, palette, layout) so CLIP text embeddings
        # accurately condition MaskGIT sketch generation.
        message = (kwargs.get("message") or "").strip()
        constraints = kwargs.get("constraints") or {}
        style_desc = []
        if constraints.get("style"):
            style_desc.append(str(constraints["style"]))
        if constraints.get("palette") and isinstance(constraints["palette"], list):
            pal_str = ", ".join(str(c) for c in constraints["palette"] if c)
            if pal_str:
                style_desc.append(f"colors: {pal_str}")
        if constraints.get("layout_hints"):
            hints_str = str(constraints["layout_hints"]).strip()
            if hints_str:
                style_desc.append(f"layout: {hints_str}")

        constraint_str = "; ".join(style_desc)
        if message and constraint_str:
            prompt_text = f"{message} ({constraint_str})"
        elif message:
            prompt_text = message
        elif constraint_str:
            prompt_text = f"UI design ({constraint_str})"
        else:
            prompt_text = "UI design"

        def _run_sync():
            import torch

            # Fallback of 768 (CLIP ViT-L/14's real text-embedding dim),
            # not the stale 4096 placeholder this used to say — only
            # matters if a loaded checkpoint's saved config is somehow
            # missing `prompt_dim` entirely (every real checkpoint from
            # the current training configs carries prompt_dim=768), but
            # a fallback should degrade to the actually-correct value,
            # not an arbitrary one — see model.py's matching fix.
            prompt_dim = getattr(self._model.module.cfg, "prompt_dim", 768)
            device = next(self._model.module.parameters()).device
            guidance_scale = self.guidance_scale

            if self._text_embedder is not None:
                prompt_embedding = self._text_embedder.embed_text(prompt_text).to(device)
                if prompt_embedding.shape[-1] != prompt_dim:
                    log.warning(
                        "sketch_prompt_dim_mismatch",
                        extra={"embedder_dim": prompt_embedding.shape[-1], "checkpoint_prompt_dim": prompt_dim},
                    )
                    prompt_embedding = torch.zeros(1, prompt_dim, device=device)
                    guidance_scale = 0.0
            else:
                prompt_embedding = torch.zeros(1, prompt_dim, device=device)
                guidance_scale = 0.0

            return self._model.sample(
                prompt_embedding,
                num_rounds=self.num_rounds,
                prior_tokens=prior_tokens,
                guidance_scale=guidance_scale,
            )

        result = await asyncio.to_thread(_run_sync)
        # DesignState.sketch_tokens.vq_tokens (§5.1) is typed as a string
        # reference, the same convention as finalize_output.image_ref — NOT
        # the raw token list. Save it as a blob and return the ref, so
        # callers (flows.py, sketch_handoff.py) treat "which tokens are
        # current" the same way they already treat "which image is current".
        tokens_ref = blobs.save_tokens(result["tokens"], prefix="sketch")
        confidence_ref = blobs.save_tokens(
            [round(c, 4) for c in result["confidence_map"]], prefix="confmap"
        )
        return {
            "tier": self.spec.tier.value,
            "vq_tokens_ref": tokens_ref,
            "confidence_map_ref": confidence_ref,
        }
