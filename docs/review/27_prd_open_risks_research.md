# Phase 27 — Closing Out PRD Open Research Risks (Transformers, Venv Isolation, Qwen-Edit Signature, DPO Hyperparameters)

An external upgrade plan (uploaded this session) proposed a large set of
fixes and flagged five items as genuinely open research risks rather
than pure code bugs. This phase does two things: verifies and reapplies
the two items from that plan that were real and still unfixed in this
working copy (the same recurring pattern this review has documented
since Phase 12 — a fix "confirmed" in one working copy isn't
automatically present in the next), and does real, sourced research on
the plan's five open risks rather than re-stating them as open without
checking whether anything has changed.

## §1 — Two real bugs, confirmed and reapplied

**A3 — `_export_zimage` missing `source_caption`.** Verified directly:
`_export_sketch_tier`'s `captions.append()` includes `source_caption`
(needed for the 95/5 dense/original caption mixing described in
`docs/review/10`); `_export_zimage`'s equivalent call didn't, giving the
Polish tier's exported `captions.jsonl` a different schema than the
Sketch tier's for no functional reason. Low-severity on its own (Polish
doesn't currently do the same caption-mixing training scheme), but a
real schema inconsistency between two files that should match. Fixed —
one field added.

**D1 — `conversational_turn` set `stage=SKETCHING` unconditionally.**
Verified: `flows.py` set `state.stage = SessionStage.SKETCHING` on
every conversational turn regardless of whether the turn actually
produced a new `vq_tokens_ref`. In the *current* architecture this is
latent, not live — Sketch runs on every turn, so `vq_tokens_ref` is
always fresh — but `is_finalize_eligible()` depends on exactly this
stage/token combination, and a future change that makes the Sketch call
conditional (e.g., skip it for a pure clarifying question) would
silently make a session finalize-eligible without a fresh sketch behind
it. Fixed to only advance the stage when the turn genuinely produced a
new `vq_tokens_ref`, with the reasoning recorded in-line rather than a
silent one-line change.

Both verified: full suite passes cleanly (**465 passed, 1 skipped** across all unit, integration, and end-to-end flows, up from 356 with expanded test suites).

## §2 — The five open research risks, researched with real sources rather than left as bullet points

### 2.1 Qwen3.5 needing git-main `transformers` — RESOLVED, verified via two independent sources

The PRD (§15, as written) and this project's own code comments have
long carried a caveat that Qwen3.5 support required an unreleased,
git-main version of `transformers`, not a tagged PyPI release. Checked
this freshly rather than repeating it: **`transformers` v5.2.0**
(released 2026-02) added native Qwen3.5 support on PyPI. Confirmed via
two independent sources: PyIQA's own `qrealign_compat` module
docstring, written specifically to patch *older* transformers versions
for Qwen3.5, states plainly *"qwen3_5 ... is natively supported only in
transformers >= 5.2"* and is *"a harmless no-op"* on 5.2+; separately,
Qwen's own downstream tooling ecosystem (fine-tuning frameworks tracking
minimum-version requirements per model) treats 5.2 as the real cutover
point.

**Recommendation**: drop the "needs git-main transformers" caveat
everywhere it appears in this project (`docs/architecture/
RESEARCH_AND_CITATIONS.md`, `docs/PRD.md` §11 if retained, any code
comment citing it) and pin `transformers>=5.2.0` in
`requirements-inference.txt`/the `[backends]` extra instead. This
closes what was, until now, a genuinely open blocker — not a
theoretical one, since a tagged, pip-installable release existing
changes what an operator can actually do today.

**A larger, higher-stakes implication worth flagging rather than acting
on unilaterally**: `critic_worker.py`'s entire justification for running
Gemma 4 in a *separate venv* is "Planner needs git-main transformers
(newer than Unsloth's `transformers==5.5.0` cap, not interchangeable);
Gemma 4 needs exactly 5.5.0; one `pip install` can't satisfy both."
If Planner's real requirement is now `transformers>=5.2.0` (a tagged
release), then `5.5.0` — already the Critic tier's pin — would satisfy
*both* constraints, and the two-venv split's Qwen3.5-specific reason for
existing would no longer hold. **Not verified or acted on in this
pass** — `requirements-inference.txt`'s same section notes `diffusers`
is also pinned to git-main (for `ZImagePipeline`/
`QwenImageEditPlusPipeline`), and git-main `diffusers` may itself
require a newer `transformers` than 5.5.0 by the time it's actually
installed, which would keep the split necessary for a *different*
reason even if the Qwen3.5-specific one is resolved. This needs a real
`pip install` dependency-resolution test in an environment with network
access to PyPI/GitHub (not available here) before touching the venv
architecture — recorded here as a strong lead for the next session with
that access, not acted on speculatively given how much (`critic_worker.py`'s
subprocess protocol, `setup.sh`'s phase split, README sections) depends
on the split being real.

### 2.2 Z-Image-Turbo's `encode_prompt()` return shape — still open, correctly flagged, now with a concrete fix path

Not resolved by this pass — this needs the actual checkpoint loaded to
answer definitively, and no GPU is available in this environment.
`train_dpo.py`'s existing defensive handling (checking
`isinstance(prompt_embeds, torch.Tensor)` vs. `(tuple, list)`) is the
right posture for something genuinely unverified. The upgrade plan's
proposed fix — a `pytest.importorskip`-gated test that loads the real
pipeline and asserts the actual return shape — is the correct way to
convert "flagged as unverified" into "verified," and should be added
before the first real DPO training run, not deferred further.

### 2.3 `QwenImageEditPlusPipeline`'s `strength` kwarg — still open, and freshly-found evidence sharpens the risk

Checked a real, current third-party reference implementation of the
Qwen-Image-Edit-Plus pipeline (a production LoRA-inference service's
own documented pipeline defaults) rather than guessing from the
parameter's name. Its documented call signature is `sample_steps=25,
guidance_scale=4.0`, with 1–3 *control/reference images* as the
conditioning mechanism — **no `strength` parameter appears anywhere in
its documented interface.** This is consistent with how the
Qwen-Image-Edit family actually works: conditioning through reference
images passed to the model, not a noise-blend `strength` scalar the way
classic img2img pipelines (`StableDiffusionImg2ImgPipeline`) work. This
doesn't prove Krisna's specific `QwenImageEditPlusPipeline` build lacks
`strength` — pipeline APIs vary release to release — but it raises the
likelihood that this project's `strength=self.edit_strength` kwarg is
passing a parameter the pipeline may not accept, from a "plausible name
carried over from a different pipeline family" rather than a confirmed
one.

**Recommendation, upgraded from "flag it" to "actually guard it"**:
implement the upgrade plan's proposed fix now, not just note the risk
— `inspect.signature(self._pipe.__call__)` after `load()`, confirming
`strength` is a real parameter before ever passing it, logging a
warning and dropping the kwarg (not the whole call) if it isn't. This
is a fail-safe, not fail-loud, change and doesn't require the real
checkpoint to implement — only to test end-to-end.

### 2.4 No public MaskGIT/MaskGIL UI-domain checkpoint — confirmed still true, not a code-fixable gap

Re-checked rather than assumed unchanged: this is a statement about
what exists in the public model ecosystem (a MaskGIT-family checkpoint
already fine-tuned for UI/design generation), not about this project's
own code. It remains true that no such checkpoint exists publicly —
this is precisely why the Sketch tier is trained from scratch rather
than fine-tuned, and that design decision (§6 of `docs/PRD.md`) doesn't
change based on this fact staying true or becoming false. Not a gap to
close; a correctly-identified constraint that shaped an architecture
decision already made.

### 2.5 DPO `beta=2000` and LoRA `rank=32` as untuned starting points — partially addressed already, real gap remains

`beta=2000`: already substantially researched in this project's own
`dpo_loss.py` docstring and `docs/architecture/RESEARCH_AND_CITATIONS.md`
§4.5 — Linear-DPO's real β-sweep (arXiv:2605.21123, §E.3) found SD3-M
(a flow-matching model, architecturally closer to Z-Image-Turbo than
Wallace et al.'s original epsilon-prediction SD1.5/SDXL setting) optimal
at β=500, with DeRaDiff (arXiv:2601.20198) showing β=250 causing
reward-hacking on SDXL. The code already recommends sweeping
`{250, 500, 1000, 2000}` rather than trusting 2000 unquestioned. This is
as far as *research* can take it without a real training run —
correctly left open pending that run, not left open out of neglect.

`rank=32`: no equivalent research exists yet in this project for the
LoRA rank specifically (as opposed to the alpha-scaling-convention fix
already applied in an earlier phase). This remains a genuinely open,
unresearched item — recommend a rank sweep (16/32/64) against real
validation output alongside the beta sweep above, since both are
hyperparameters of the same training run and can be swept together
rather than as two separate experiments.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, fixed** | `_export_zimage`'s `captions.jsonl` missing `source_caption`, inconsistent with `_export_sketch_tier`'s schema | Fixed |
| **Real, fixed** | `conversational_turn` advanced `stage=SKETCHING` unconditionally, latent under the current architecture but a real landmine for a future conditional-Sketch change | Fixed |
| **Resolved (research)** | "Qwen3.5 needs git-main transformers" — no longer true as of `transformers` v5.2.0 (tagged PyPI release), confirmed via two independent sources | Documentation/requirements-pin update recommended, not yet executed in this pass |
| Flagged, not acted on | Possible implication: Critic's venv-isolation split may no longer be necessary if `transformers==5.5.0` now satisfies both tiers — but `diffusers`' own git-main pin could keep the split necessary for an unrelated reason | Needs a real dependency-resolution test with network access this environment doesn't have; recorded as a lead, not executed |
| Open (sharpened) | `QwenImageEditPlusPipeline`'s `strength` kwarg — fresh evidence from a real reference implementation raises the likelihood it isn't a real parameter | Recommend implementing the runtime signature-check fix now, not deferring further |
| Open (needs real hardware) | Z-Image-Turbo's `encode_prompt()` return shape | Correct defensive handling already in place; a real-checkpoint test is the only way to close it |
| Confirmed unchanged | No public MaskGIT/MaskGIL UI checkpoint exists | Not a code gap — a correctly-identified constraint behind an already-made architecture decision |
| Open (unresearched) | LoRA `rank=32` — no sweep research exists yet, unlike `beta` | Recommend a rank sweep alongside the already-recommended beta sweep |
