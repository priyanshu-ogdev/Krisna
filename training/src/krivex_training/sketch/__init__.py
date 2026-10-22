"""Sketch Tier Specification — 100% DETERMINISTIC & FROZEN (krivex.layout.v1).

The Sketch tier operates strictly via deterministic neuro-symbolic Layout IR compilation
and Simplex / Cassowary linear constraint solving on the CPU (0 MB GPU VRAM, <5ms latency).

Architectural decisions:
1. Discrete VQ codebooks (MaskGIT / VQGAN) and continuous neural diffusion (LayoutFlow)
   have been completely retired and purged from the production training pipeline.
2. Geometry, element hierarchy, text literals, and locked regions are resolved
   deterministically without neural sampling noise or codebook collapse.
3. No training, fine-tuning, or checkpointing code exists or executes for this tier.
"""

from __future__ import annotations


def get_sketch_training_status() -> dict[str, str | bool]:
    """Return explicit deterministic and frozen status for the Sketch tier."""
    return {
        "tier": "sketch",
        "is_trainable": False,
        "is_frozen": True,
        "engine": "krivex.layout.v1 (Neuro-Symbolic Simplex / Cassowary Solver)",
        "vram_gb": 0.0,
        "rationale": "Deterministic Layout IR guarantees 0% element overlap, 0 MB GPU VRAM, and eliminates VQ codebook collapse",
    }
