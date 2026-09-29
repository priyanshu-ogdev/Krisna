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

UPGRADE (training-data-integrity review, this pass): the
gemma_critique-sourced rows this dataset reads were, until a fix in
orchestrator/flows.py this same review pass, at real risk of a (prompt,
image) mixup — critique_pass() could run turns after the image was
actually generated and was re-synthesizing "the" prompt from whatever
the conversation had drifted to by critique time, not what was true
when the image was rendered. flows.py's finalize() now freezes the
real generation-time prompt on DesignState.finalize_output.prompt_used
and critique_pass() reads it back instead of re-deriving it — this
dataset has no way to detect or repair a mismatch in rows written
before that fix, since PreferenceStore's schema has no independent
timestamp-of-generation to cross-check `prompt` against; new rows going
forward are correct.
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
        split: str = "train",
        val_fraction: float = 0.05,
        flip_prob: float = 0.0,
    ) -> None:
        import hashlib

        import torch
        from torchvision import transforms

        from krisna_inference.backends.common import BlobStore
        from krisna_training.dpo.preference_store import PreferenceStore

        self.db_path = db_path
        self.blob_root = blob_root
        self.resolution = resolution
        self.sources = sources
        # UPGRADE (this review pass, reconsidered from a prior "flagged
        # not implemented" note): unlike the Sketch tier, this dataset
        # loads REAL PIXELS (not pre-tokenized VQ codes) — token-space
        # flipping is unsafe (a VQGAN token's embedding encodes its
        # patch's un-flipped content; rearranging token POSITIONS
        # without flipping the underlying pixel content produces a
        # geometrically-flipped-but-content-wrong image), but a real
        # pixel-space flip, applied here before ToTensor/Normalize, is
        # exactly correct and genuinely feasible for this dataset.
        # DELIBERATELY DEFAULTS TO 0.0 (no behavior change) rather than
        # a nonzero value: this is the FINAL high-fidelity output stage,
        # where a UI screenshot's on-screen text gets mirrored into
        # unreadable backward glyphs by a horizontal flip — training on
        # that risks teaching the model to occasionally render backward/
        # garbled text, arguably a worse failure mode here than in the
        # Sketch tier (low-fidelity draft) that finding #7 flagged the
        # same caveat for. Real data-forge metadata for per-record text/
        # OCR density (which would let this be applied only to
        # low-text-density UI screenshots) isn't available on
        # PreferencePair's schema — so this ships correctly implemented
        # and opt-in via flip_prob, rather than silently defaulting on.
        self.flip_prob = flip_prob

        self.store = PreferenceStore(db_path=db_path)
        self.blobs = BlobStore(root=blob_root)
        all_pairs = self.store.list()

        if sources:
            all_pairs = [p for p in all_pairs if p.source in sources]

        # UPGRADE (validation loop, finding #3): the DPO stage had no
        # heldout signal at all — only training loss on a preference
        # margin, which for DPO can shrink purely by reward-hacking
        # (drifting away from the reference distribution) rather than by
        # genuinely improving preference alignment. Deterministic
        # hash-of-id split (not random.random()) so the same pair always
        # lands in the same split across process restarts and DataLoader
        # workers, matching the PRD's own 5%-stratified-holdout
        # convention (§9's data-forge s09_heldout) rather than inventing
        # a different holdout fraction for this stage.
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

        def _is_val(pair_id: str) -> bool:
            digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
            return (int(digest[:8], 16) / 0xFFFFFFFF) < val_fraction

        if split == "val":
            self.pairs = [p for p in all_pairs if _is_val(p.id)]
        else:
            self.pairs = [p for p in all_pairs if not _is_val(p.id)]

        if not self.pairs:
            raise ValueError(
                f"No preference pairs found in {db_path} for sources={sources} split={split!r}. "
                "Run python scripts/data-forge/sync_to_training.py first."
            )

        log.info(
            "dpo_dataset_loaded",
            extra={
                "pairs": len(self.pairs), "sources": sources or "all", "resolution": resolution,
                "split": split, "total_available": len(all_pairs),
            },
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

        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            # UPGRADE: a SINGLE shared coin-flip applied to BOTH images —
            # not two independent draws. DPO measures a RELATIVE
            # preference (chosen vs rejected) for the SAME prompt/state;
            # mirroring both sides together preserves that relative
            # judgment (a mirrored comparison of two mirrored designs is
            # still the same comparison), whereas flipping them
            # independently would compare two images under different,
            # unrelated geometric transforms and corrupt the pair's
            # actual preference signal.
            from torchvision.transforms import functional as TF

            chosen_img = TF.hflip(chosen_img)
            rejected_img = TF.hflip(rejected_img)

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
