"""Sketch tier training — closes the gap flagged throughout this project's
README: `inference/sketch_backend.py` and `inference/maskgit_model.py`
implement the architecture *contract* and inference-time sampler, but
until now there was no training code to actually produce a checkpoint that
satisfies that contract. This package is that training code.

This is a genuinely separate concern from `inference/maskgit_model.py`,
which stays as-is: that module defines the sampling contract a checkpoint
must satisfy (`.forward(tokens, mask, prompt_embedding) -> (logits,
critic_scores)`); this package's `model.py` is one concrete architecture
that satisfies it, plus everything needed to train it.

Same lazy-import discipline as inference/ and verifiers/: torch is only
imported inside functions/methods, never at module scope, so importing
this package doesn't require torch to be installed.
"""
