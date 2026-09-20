"""Dataset implementation for Diffusion-DPO preference training.

Loads preference pairs from the SQLite PreferenceStore, resolves
blob:// image references through BlobStore, and applies standard
normalization and square center-cropping.

P2b prompt augmentation (semantic audit): `__getitem__` mixes in a
constraint-derived prompt format 30% of the time alongside the raw
Pick-a-Pic v2 / HPDv2 prompt. This closes the training/inference
distribution gap — inference sends prompts like
"dark dashboard for crypto app, style: dark, palette: #1A1A2E"
(synthesized by flows._synthesize_dpo_prompt), while Pick-a-Pic
prompts look like "vibrant sunset, digital art". A 30% mix rate is a
starting point; tune based on real DPO training quality metrics.
"""

from __future__ import annotations

import logging
import random
from typing import Any

log = logging.getLogger("krisna_training.polish.dpo_dataset")

# P2b: fraction of training steps to substitute a constraint-style prompt
# for the raw dataset prompt, approximating inference-time prompt format.
_CONSTRAINT_PROMPT_AUG_PROB = 0.30


def _constraint_style_prompt(pair) -> str:
    """Build a prompt in the inference format from whatever structured
    metadata the PreferencePair carries. Falls back gracefully when fields
    are absent — Pick-a-Pic / HPDv2 pairs carry no UI-domain metadata, so
    this will usually produce a minimal string; DesignSense / DesignPref
    pairs may carry richer tags."""
    parts = []
    style = getattr(pair, "style", None) or getattr(pair, "tag", None)
    if style:
        parts.append(f"style: {style}")
    # Use raw prompt words as the intent anchor when no style tag is available
    if not parts and pair.prompt:
        parts.append(pair.prompt[:80])  # trim to ~intent length
    return ", ".join(parts) if parts else "High quality UI design"


class PreferencePairDataset:
    """PyTorch Dataset yielding (chosen_pixel_values, rejected_pixel_values, prompt)

    from real human preference pairs recorded in the PreferenceStore.
    """

    def __init__(
        self,
        db_path: str,
        blob_root: str,
        sources: list[str] | None = None,
        resolution: int = 512,
    ) -> None:
        import torch
        from torchvision import transforms

        from krisna_inference.backends.common import BlobStore
        from krisna_training.dpo.preference_store import PreferenceStore

        self.db_path = db_path
        self.blob_root = blob_root
        self.resolution = resolution
        self.sources = sources

        self.store = PreferenceStore(db_path=db_path)
        self.blobs = BlobStore(root=blob_root)
        self.pairs = self.store.list()

        if sources:
            self.pairs = [p for p in self.pairs if p.source in sources]

        if not self.pairs:
            raise ValueError(
                f"No preference pairs found in {db_path} for sources={sources}. "
                "Run python scripts/data-forge/sync_to_training.py first."
            )

        log.info(
            "dpo_dataset_loaded",
            extra={"pairs": len(self.pairs), "sources": sources or "all", "resolution": resolution},
        )

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        chosen_img = self.blobs.load_image(pair.chosen_ref).convert("RGB")
        rejected_img = self.blobs.load_image(pair.rejected_ref).convert("RGB")
        # P2b prompt augmentation: 30% of the time substitute a constraint-style
        # prompt (approximating inference-time format) for the raw dataset prompt.
        # This prevents the LoRA from overfit-optimizing for the Pick-a-Pic v2 /
        # HPDv2 prompt distribution, which doesn't match what flows.py sends.
        if random.random() < _CONSTRAINT_PROMPT_AUG_PROB:
            effective_prompt = _constraint_style_prompt(pair)
        else:
            effective_prompt = pair.prompt
        return {
            "chosen_pixel_values": self.transform(chosen_img),
            "rejected_pixel_values": self.transform(rejected_img),
            "prompt": effective_prompt,
        }
