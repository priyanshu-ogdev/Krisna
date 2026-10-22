"""Krisna Training Package — Single-Model Diffusion Polish Adaptation.

Architectural Contract (PRD §6):
Krisna operates on a strict single-trainable-tier paradigm:
1. Tier 1 (Planner): 100% FROZEN foundation model (Qwen3.5-9B GGUF via llama.cpp)
   grounded via UICrit RAG and pushdown automata CFG grammar logit constraints.
2. Tier 2 (Sketch): 100% DETERMINISTIC neuro-symbolic Layout IR engine (krisna.layout.v1)
   executing simplex constraint solving on CPU (0 MB GPU VRAM, <5ms latency).
3. Tier 3 (Polish): The ONLY trainable generative model in the system.
   Trains supervised QDoRA/LoRA adapters, DPO preference alignment, and Consistency
   Flow Matching (CFM) few-step fast preview distillation for Qwen-Image-2.1.
"""

from __future__ import annotations

__version__ = "0.2.0"

ACTIVE_TRAINABLE_TIERS = ("polish_default", "polish_dpo", "polish_cfm")
FROZEN_TIERS = ("planner", "sketch", "critic")
