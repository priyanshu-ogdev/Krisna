"""Verifier stack — PRD §6: "CLIP/SigLIP, OCR, layout-IoU, aesthetic/safety,
handoff-consistency. <1B combined. Off-the-shelf, minimal training."

Design note on residency: unlike the swappable tiers (planner/sketch/polish/
critic), the verifier stack is NOT wired into SwapOrchestrator's
ResidencyState machine. §5.3's Finalize sequence has it running WHILE the
Polish tier is still resident ("Polish Tier generates -> Verifier Stack
scores output -> Orchestrator unloads Polish Tier"), and at <1B params
combined it's cheap enough to just stay loaded permanently as a small fixed
VRAM overhead (~1-2GB) rather than participate in swap semantics at all.
See verifier_stack.py's VERIFIER_VRAM_RESERVED_GB.

Everything here follows the same lazy-import discipline as
krisna_inference (formerly krisna_orchestrator)/inference/: torch/transformers/opencv/easyocr are only
imported inside methods, never at module scope.
"""
