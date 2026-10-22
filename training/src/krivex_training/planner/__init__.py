"""Planner Tier Specification — 100% FROZEN (PRD §6 No-RLHF Architectural Revision).

The Planner operates strictly as a frozen Vision-Language-Action (VLA)
foundation model (Qwen3.5-9B-Instruct GGUF via llama.cpp with W4A16 / AWQ quantization)
grounded via:
1. RAG retrieval over UICrit (krivex_inference.backends.planner_rag) providing
   empirical human design rules without parameter modification.
2. Pushdown automata Context-Free Grammar (CFG) logit masking guaranteeing 100%
   syntactically valid JSON Schema compliance (`krivex.delta.v1`).

No fine-tuning, LoRA adapters, or RLHF loops are trained for this tier.
All legacy LoRA training scripts have been purged.
"""

from __future__ import annotations


def get_planner_training_status() -> dict[str, str | bool]:
    """Return explicit frozen status for the Planner tier."""
    return {
        "tier": "planner",
        "is_trainable": False,
        "is_frozen": True,
        "runtime": "llama.cpp (Qwen3.5-9B-Instruct GGUF)",
        "grounding": "UICrit RAG + Context-Free Grammar (CFG) logit constraints",
        "rationale": "PRD §6: Frozen base model prevents conversational drift and eliminates AI-judge RLHF loops",
    }
