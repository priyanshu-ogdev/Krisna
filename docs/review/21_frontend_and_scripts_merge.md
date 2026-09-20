# Phase 21 — Frontend/Scripts Merge, Citation Verification, and Two More Recurring-Regression Instances

This phase merges two divergent working copies (an upload with extensive
training/inference fixes — Sketch CFG, gradient checkpointing, the DPO
LR scheduler, the Planner JSON-delta bug, the Finalize crash bug — and
this session's own newer frontend work — the pixel-forming canvas, phase
pipeline, live GPU/RAM hardware utilization) rather than picking one and
discarding the other. Also reviews the root-level scripts and verifies
the Muse CFG citation directly against its primary source.

---

## §1 — Merge: which side had what, kept both

Diffed the two working copies file-by-file before merging anything.
Confirmed by direct inspection (not assumed from either side's own
summary):

- **The upload's frontend was strictly older** than this session's —
  it had the critique-panel restore (verified identical, word-for-word
  comment included, to this session's own earlier fix) but not the
  phase-pipeline SVG, pixel-forming canvas, or live-hardware
  utilization display built afterward in this session. Took this
  session's frontend files wholesale (`index.html`, `app.js`,
  `style.css`, `server.js`) as the base going forward.
- **This session's backend/training fixes were the older side** —
  the upload had real work (see its own summary) this session's copy
  never had. Took the upload's `inference/`, `training/`, `data-forge/`
  trees as the base.
- Added this session's own `/session/{id}/render` endpoint and
  `real_vram`/`real_ram` hardware-probe fields to the upload's
  `service.py` (neither existed there) — these are what the merged
  frontend's pixel-forming reveal and live-hardware bars actually call.
  Ported the matching tests over; both pass against the merged tree.

## §2 — Two more real bugs found in the upload's own tests, independent of the merge

Not caused by merging — found while running the upload's own test suite
as-is, before any of this session's changes touched it.

**Bug 1 — a restored regression test re-encoded a misconception this
review already corrected once.** `test_critic_low_vram_tier_has_real_
headroom_not_exact_equality` (added by whatever process reapplied the
Critic VRAM fix in the uploaded session) asserted:

```python
baseline_vram = LOW_VRAM_REGISTRY[Tier.PLANNER].vram_gb + LOW_VRAM_REGISTRY[Tier.SKETCH].vram_gb
assert baseline_vram + critic_spec.vram_gb < 12.0
```

i.e., Planner + Sketch + Critic must all fit a 12GB envelope
*simultaneously*. That's not how this system admits a swapped-in tier —
baseline is unloaded *before* Critic (or any Polish tier) is admitted,
per `swap_orchestrator.py`'s real admission sequence. This is the exact
same misconception Phase 6 already found and retracted as a "critical
bug" earlier in this review (the baseline-plus-candidate sum was never
the real admission check). The sibling test one function above in the
same file already asserts the correct thing — each swappable tier fits
*alone*, baseline fits *alone*, never summed. Removed the incorrect
final assertion; the test now checks only what it can actually claim
(Critic's own real per-tier headroom), with a comment explaining why the
removed line was wrong rather than silently dropping it.

**Bug 2 — a stale value in a different test file.**
`test_low_vram_env_var_sets_critic_max_gpu_gb` asserted
`backend.max_gpu_gb == 12.0` — `factory.py`'s actual live default is
already `11.5` (the fixed value). Updated the assertion to match.

Both are now-fixed, both verified against the real merged code: full
`tests/inference` suite passes (136 passed, 1 skipped — the skip is
pre-existing and unrelated).

## §3 — The `data-forge/orchestrator.py` docstring/numbering bug, found regressed again

Checked whether this session's earlier fix (module docstring describing
the real 6-phase order after the escalation-ordering fix; the duplicate
"Phase 5" label corrected to "Phase 6") survived into the uploaded
working copy. **It hadn't** — the uploaded copy had the underlying
escalation-ordering *code* fix (Phase 3 escalation correctly runs before
Phase 4 recaption/structure — the actual bug fix, and the more important
one), but the docstring still described the old order, and the
numbering collision was still present. Reapplied both corrections to
the merged tree. Not itself a functional bug (nothing reads the
docstring or a comment's phase number at runtime) — but exactly the
class of drift this whole review exists to catch, and worth fixing on
sight rather than leaving for a fourth session to rediscover.

## §4 — Root scripts: reviewed, one real gap found and fixed

`setup.sh`, `run_data_forge.sh`, `run_inference.sh`, `train.sh` — all
four pass `bash -n` syntax checks. `train.sh --list` runs cleanly and
its per-tier design-intent summaries were spot-checked against the real
current YAML configs and backend code (not just re-trusted from the
uploaded session's own claim of having verified them) — accurate.

**Gap found:** `setup.sh` had no phase for `inference-frontend/` at all
— a real, working piece of this project that a fresh operator running
`./setup.sh` would have no way of discovering existed. Added a
`frontend` phase (`npm install`, with a clear "Node not found" error
message rather than a cryptic `npm: command not found`), deliberately
**not** added to the default phase list — the backend runs fully
without it (any HTTP client can talk to the FastAPI service directly),
so a Node/npm requirement shouldn't be forced on a Python-only setup.
Added a one-line mention in the "Next steps" banner so it's discoverable
without reading the script's own header comment.

## §5 — Citation verification: Muse's CFG formula, checked against the primary source directly

`maskgit_model.py`'s CFG implementation cites Chang et al. 2023 ("Muse:
Text-To-Image Generation via Masked Generative Transformers"), §2.7,
for both the guidance formula and the 10% training-time dropout rate.
Fetched the actual paper (arXiv:2301.00704) rather than trusting the
docstring's citation on faith — every specific claim checks out exactly:

- **Formula**: paper states `ℓ_g = (1+t)ℓ_c - t·ℓ_u` in §2.7, equation
  (1). The code's `logits = (1.0 + t_step) * logits - t_step * uncond_logits`
  is a direct, correct implementation of this.
- **10% dropout**: paper states *"we remove text conditioning on 10% of
  samples chosen randomly"* at training time — matches this project's
  own `cfg_dropout_prob=0.1` default exactly, confirmed as the same
  design choice, not a coincidental match.
- **Linear guidance ramp**: paper states guidance is linearly increased
  through the sampling procedure rather than held constant, "to reduce
  the hit to diversity." The code's `t_step = guidance_scale * (step /
  num_rounds)` implements exactly this ramp.

**No correction needed — this is an accurate, precise citation**,
verified line-by-line against the primary source rather than assumed
correct because it sounded plausible.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| Merge (not a bug) | Two working copies had diverged, each with real work the other lacked | Merged deliberately, both sides preserved, verified via full test suite |
| **Real, fixed** | A restored regression test re-asserted a misconception (baseline+candidate VRAM summed) that Phase 6 already retracted once | Fixed — test corrected to match the real admission model |
| **Real, fixed** | A stale `max_gpu_gb == 12.0` assertion, post-fix value is `11.5` | Fixed |
| **Real, fixed (recurrence)** | `orchestrator.py`'s module docstring + duplicate phase numbering, previously fixed in this session, had reverted in the uploaded copy | Fixed again |
| **Real, fixed** | `setup.sh` had no path to discovering or installing `inference-frontend/` | Added `frontend` phase + next-steps mention |
| Info | Muse CFG citation (formula, §2.7, 10% dropout, linear ramp) | Verified accurate against the primary source, no correction needed |

## On the recurring-regression pattern itself

This is at least the fifth distinct instance across this review of a
fix reverting between working copies (`maskgit_vq`, the Critic VRAM
figures, the safety-gate wiring doc, the escalation-ordering docstring
— twice now — and the two test bugs in §2). The pattern is upstream of
this review — it's about how working copies get edited and re-uploaded
outside this conversation, not a flaw in any individual fix. Restated
plainly, again: treat this merged archive as the thing to diff against
next time, not something to assume is still intact after further
editing elsewhere.
