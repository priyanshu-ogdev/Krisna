"""Real model-loading backends for each PRD §6 tier.

Everything in this package is imported lazily (inside load(), not at module
scope) so that `krisna_inference (formerly krisna_orchestrator)` core (orchestrator, state machine,
service) never requires torch/diffusers/transformers/unsloth to be
installed — those stay optional, GPU-machine-only dependencies. See
factory.py for how these plug into SwapOrchestrator via the same
ModelBackend interface MockBackend already satisfies.
"""
