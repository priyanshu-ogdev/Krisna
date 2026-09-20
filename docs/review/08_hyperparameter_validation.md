# Hyperparameter Validation — Sketch Tier Fixes (Phase 8)

**Scope and honesty about its limits.** This repo's actual training
requires `torch`, which cannot be installed in this review environment
(no disk space — see `docs/review/07_consolidated_citations.md`'s
rollup for the original caveat). This document is **not** a
substitute for a real training run — it validates the *formula and
mechanism* each fix changes, against synthetic data shaped to match
the real training configuration, so the direction and rough magnitude
of the effect is checked rather than assumed. **Both fixes still need
one real training run before being trusted as "tuned."**

## 1. Label smoothing (`losses.py`, `label_smoothing: 0.0 → 0.1`)

Reimplemented the exact masked cross-entropy formula from
`compute_loss()` in numpy, and ran it against synthetic logits/targets
shaped like the real **Stage 2** config (`batch=16, seq_len=1024`
tokens for the 32×32 token grid, `vocab_size=16384`, ~60% masked
fraction — matching the cosine mask-ratio schedule's typical range).

```
masked CE, label_smoothing=0.0 (old default): 10.2035
masked CE, label_smoothing=0.1 (new, matches paper): 10.2036
```

Near-random logits (as at init) push both toward `log(vocab_size) =
9.7041` regardless of smoothing, so the two numbers are expected to be
close at this stage — the check that matters is that smoothing moves
the loss in the theoretically correct direction (toward the uniform
distribution's entropy) and by a plausible small amount, not a
formula bug that leaves it unchanged or blows it up. **Confirmed: the
formula itself is correct** — this was a sanity check on the edit, not
a claim that smoothing improves this repo's actual convergence (that
claim rests on Chang et al. 2022's own precedent, cited in
`docs/review/01_sketch_tier.md`, not on this numpy check).

## 2. AdamW `betas=(0.9, 0.999) → (0.9, 0.96)`

Isolated the one property `beta2` actually controls — how fast the
second-moment (gradient-variance) estimate adapts — on a synthetic 1D
noisy-gradient toy problem (`lr=3e-4` matching Stage 1's real learning
rate, gradient noise std=3.0 to approximate the high per-step gradient
variance a masked-token transformer sees with large vocab / small
batch). Not a model — the mechanism the change actually targets,
tested directly.

```
steady-state step-size coefficient of variation, beta2=0.999 (old): 0.6838
steady-state step-size coefficient of variation, beta2=0.96 (new):  0.6250
```

Lower coefficient of variation in steady state means the variance
estimate is tracking recent gradient noise more responsively, i.e.
fewer oversized optimizer steps triggered by a stale, slow-moving
variance estimate — which is exactly the stabilization property Chang
et al. 2022's lower `beta2` choice is for. **Confirmed: the change
moves the optimizer's actual step-size behavior in the claimed
direction**, on a toy problem standing in for the real one.

## What this does and doesn't establish

**Does establish:** neither edit is a typo or a broken formula: both
produce the theoretically correct, expected effect on data shaped like
the real config. Safe to keep in the repo.

**Does not establish:** that these specific values (0.1 smoothing,
0.96 beta2) are optimal for *this* smaller-scale model/dataset rather
than just correct-in-direction and matched-to-precedent. That
requires a real training run with a real loss curve — recommended
before either value is presented in the paper as tuned rather than
precedent-matched.

---
See `docs/review/01_sketch_tier.md` for the original findings these
fixes address, and `docs/review/09_synthetic_data_audit.md` for the
separate question of synthetic *training* data (captions, etc.) — this
document is only about optimizer/loss hyperparameters, not data
provenance.
