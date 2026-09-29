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
    def __init__(self, spec, checkpoint_path: str | None = None, num_rounds: int = 8) -> None:
        super().__init__(spec)
        self.checkpoint_path = checkpoint_path
        self.num_rounds = num_rounds
        self._model = None

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

        def _run_sync():
            import torch

            # Real prompt conditioning would come from the planner's hidden
            # states / a shared embedding space — placeholder zeros here
            # since that cross-tier embedding contract is a training-time
            # decision this backend doesn't own. Sized from the loaded
            # checkpoint's own config (training/sketch/model.py's
            # SketchModelConfig.prompt_dim), not a hardcoded guess — a
            # checkpoint trained with a different prompt_dim (e.g. CLIP's
            # 768 instead of the placeholder default) still loads correctly.
            prompt_dim = getattr(self._model.module.cfg, "prompt_dim", 4096)
            prompt_embedding = torch.zeros(1, prompt_dim, device=next(self._model.module.parameters()).device)
            return self._model.sample(
                prompt_embedding, num_rounds=self.num_rounds, prior_tokens=prior_tokens
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
