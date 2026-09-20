# Phase 17 — Sketch-Tier Inference Conditioning & Classifier-Free Guidance

Follow-up to `10_synthetic_data_generalization_fix.md`. That doc closed
two training-side gaps (caption-style mixing, CFG conditioning dropout)
and reported both as tested and working — true, but neither fix had been
checked against the actual inference call path. Once checked, two
deeper problems surfaced.

## What was found

**1. No text ever reached the Sketch model at inference, at all.**
`sketch_backend.py`'s `run()` sent `prompt_embedding = torch.zeros(1,
prompt_dim, ...)` **unconditionally, on every call**, regardless of what
the user typed. Disclosed as a placeholder in the original comment, not
a hidden bug — but it meant the caption-mixing fix in Phase 10 had
nothing to act on at inference: there was no user text flowing into the
model to be either dense-caption-style or short-caption-style.

**2. The CFG-dropout training investment was entirely unused.**
`maskgit_model.py`'s `sample()` ran a single conditional forward pass per
step — no unconditional forward pass, no guidance-scale combination,
nowhere. The model was trained with 10% conditioning dropout specifically
so classifier-free guidance would be usable at inference; nothing ever
used it.

## The fix

**(a) Real text conditioning** (`sketch_backend.py`): `load()` now also
loads `krisna_inference.verifiers.common.get_clip_embedder()` — the same
singleton `sketch/train.py`'s `collate_fn` uses, guaranteeing the
embedding space matches by construction. `run()` embeds the user's actual
`message` (already flowing through `**kwargs` from
`SwapOrchestrator.run_conversational_turn`, previously silently ignored).
Empty/missing messages fall back to embedding the literal string `"UI
design"` — matching `make_collate_fn`'s own fallback for captionless
records — rather than a zero vector, which is now reserved specifically
for CFG's unconditional branch. A prompt-dimensionality mismatch between
the loaded embedder and the checkpoint's expected `prompt_dim` falls back
to zero-embedding + disabled guidance rather than guessing a projection
that was never trained.

**(b) Real classifier-free guidance** (`maskgit_model.py`): `sample()`
gained a `guidance_scale` parameter (default `0.0`, exact prior behavior,
so existing callers are unaffected), implementing the real published
formula — researched, not assumed — against Muse (Chang et al. 2023,
"Muse: Text-To-Image Generation via Masked Generative Transformers",
§2.7), the actual T2I MaskGIT-lineage paper this architecture already
follows, whose 10% training-time conditioning-dropout rate matches this
project's own `cfg_dropout_prob` default exactly:

    l_g = (1 + t) * l_c - t * l_u

linearly ramped from 0 to `guidance_scale` across sampling rounds per
Muse's own reported refinement. `uncond_embedding` defaults to
`torch.zeros_like(prompt_embedding)` — the same zero-vector convention
training's `collate_fn` uses for CFG-dropped samples. `guidance_scale`
defaults to `3.0` on `SketchBackend`, documented as a moderate
literature-anchored starting point pending real tuning (no trained
checkpoint exists yet), env-overridable via `KRISNA_SKETCH_GUIDANCE_SCALE`.

## Verification

No GPU/`torch` in this environment, so verification used a minimal
numpy-backed tensor stub reproducing real torch's tensor/softmax/no_grad
surface, driving the actual `MaskGITSketchModel.sample()` code end to
end (not a rewrite of it):

- **Guidance formula correctness**: verified numerically — as `t`
  increases 0→5, the margin favoring the conditional class over the
  unconditional one grows monotonically (0.2 → 2.4 → 6.8 → 11.2),
  matching Muse's described behavior exactly.
- **Backward compatibility**: `guidance_scale=0.0` calls the module
  exactly once per round — identical to pre-fix behavior.
- **Guidance is real, not inert**: with a fake module returning
  genuinely different logits for cond vs. uncond calls, `guidance_scale=5.0`
  demonstrably shifted sampled tokens toward the conditional preference.
- **Prompt routing**: real message embedded verbatim; empty message
  embeds `"UI design"`, not zeros; missing/unavailable embedder falls
  back to zero embedding with `guidance_scale` forced to `0.0`.

Permanent tests: `tests/training/test_maskgit_cfg.py` (formula,
backward compatibility, uncond-embedding default, output actually
changes under guidance) and `tests/inference/test_sketch_backend_prompt.py`
(message routing, fallback text, no-embedder/dim-mismatch degradation).

## What this means for Phase 10's caption-style question

Phase 10 flagged a residual risk: even with caption-mixing fixed, the
model still spends 95% of training exposure on dense VLM-style captions,
while real users type short, plain prompts closer to the 5%
`source_caption` style. That risk is real and — per Betker et al. 2023's
own rationale for the 95/5 ratio existing at all — not something to fix
further; the 5% exposure is deliberately sized to generalize to exactly
that real-world case without giving up the dense-caption quality
benefit. What this phase adds is the missing precondition for that 5%
exposure to matter at all: until now, no user prompt reached the model
in any style, dense or short.

## Regression note (added later in this review)

This fix (both `sketch_backend.py`'s real conditioning and
`maskgit_model.py`'s CFG implementation, plus their two test files) was
found completely missing from a later working copy in this review — not
a partial drift, the whole feature reverted to its pre-fix state,
including `factory.py`'s `guidance_scale` env-var wiring. Caught while
verifying citations for an unrelated task, by grepping for this doc's
own claimed Muse citation and finding zero hits in the actual source
file. Reapplied and re-verified against the real code using the same
numpy-backed torch-stub technique described above. This is at least the
fourth distinct instance of previously-fixed code reverting in an
uploaded working copy over the course of this review — see
`docs/review/README.md`'s standing note on this pattern.
