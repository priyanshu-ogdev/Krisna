# 36 — Model 4/5 deep review: Polish-Quality (Qwen-Image-Edit-2511)

This tier ships frozen (PRD §6 — no LoRA, no training pipeline), so
this review covers the inference backend and its connections to Models
1-3 only. No data-forge/training-side review applies.

## Critical finding: `edit_strength` has never actually worked — now definitively confirmed, not just uncertain

`QwenImageEditBackend` already had a defensive runtime check (B3) that
inspects the loaded pipeline's real `__call__` signature and sets
`edit_strength = None` if `strength` isn't a parameter, logging a
warning and skipping it rather than raising `TypeError`. This was
previously documented as version-dependent uncertainty ("whether the
Edit-Plus variant exposes it depends on the diffusers version... verify
the correct kwarg name").

This review resolved that uncertainty for real, twice over:

1. Checked out the **exact commit** `requirements-inference.txt` pins
   (`cc8644b447d8f11074d3df06d0ee0e3e7c91bf75`) and read
   `QwenImageEditPlusPipeline.__call__`'s real signature directly:
   `strength` appears **zero times**.
2. Independently confirmed against the installed PyPI `diffusers==0.40.0`
   package in this review's own environment: same result,
   `'strength' in sig.parameters` is `False`.

**Practical consequence**: B3's check fires every single time against
this project's actual pin — `edit_strength` has never had any effect on
a real run. This isn't a bug in the detection logic itself, which
correctly, defensively discovers this and degrades gracefully. It's a
previously-undocumented *fact* about this pipeline's real API surface,
mischaracterized in the code as an open question rather than a known,
confirmed non-functional state.

**Root cause, now understood**: Qwen-Image-Edit is an
instruction-based edit pipeline (architecturally closer to
InstructPix2Pix/FLUX-Kontext) rather than an SDEdit partial-noise
pipeline. "How much the output changes" is controlled through the edit
instruction text and `true_cfg_scale`, not a numeric strength
parameter — which is likely why this pipeline family never grew one at
all, not a temporary omission.

## What was fixed

- Rewrote `edit_strength`'s constructor docstring, the B3 check's
  comment and log message, the module-level docstring, and
  `_edit_instruction_from_constraints`'s comment — all previously said
  or implied a mask/strength-based mechanism was real or
  version-uncertain; all now state the confirmed fact plainly. Kept the
  parameter and the runtime check (not removed) specifically because
  it's forward-compatible: a future diffusers version that adds
  `strength` support would be picked up automatically, with zero code
  changes needed here.
- Added `tests/inference/test_polish_quality_backend.py` — this backend
  had **no direct test coverage at all** before this pass, despite the
  B3 logic being exactly the kind of subtle, silently-breakable behavior
  that deserves one. 11 new tests: both directions of the B3 detection
  (strength absent → None; strength present → forward-compat preserved,
  using a real `inspect.signature()`-compatible fake `__call__`, not a
  bare MagicMock that would trivially pass either way), `run()`'s
  `handoff_image_ref`/not-loaded guards, confirming `strength` is
  correctly included/omitted from the real pipeline call based on
  `edit_strength`'s state, and `_edit_instruction_from_constraints`'s
  text construction including locked-region instructions.

## `num_inference_steps=40` — not a deviation, a previously-uncited correct choice

Noticed `num_inference_steps=40` differs from
`QwenImageEditPlusPipeline.__call__`'s own bare signature default of 50,
with no comment explaining the choice. Researched rather than assumed
either direction: two independent sources confirm 40 is the real,
official Qwen-Image-Edit-2511 model-card setting — the official w3ss
GGUF model card's own usage snippet (`"num_inference_steps": 40`), and a
hosted-API provider's docs explicitly noting "the upstream model card
demonstrates 40 steps with true_cfg_scale 4.0" (also matching this
backend's `true_cfg_scale=4.0`, independently confirmed equal to the
pipeline's own default). The pipeline class's bare `50` is a generic
Python fallback, not a per-checkpoint recommendation. Both values here
were already correct — added the citation that was previously missing,
not a value change. Worth recording as a reminder that "differs from the
library default" is not itself evidence of a bug; it's a prompt to go
verify which one is actually right for the specific checkpoint in use.

## Connection to Model 1 (Planner): verified already correct, no fix needed

Checked whether this tier was missing the same Planner-reasoning-note
enrichment found and fixed for Sketch (`docs/review/32`) and
Polish-Default (`docs/review/35`). Traced `flows.py::finalize()`:
`prompt=effective_prompt` is passed uniformly to
`orchestrator.request_finalize()` regardless of which Polish tier ends
up handling the request (`preferred = Tier.POLISH_QUALITY if quality
else Tier.POLISH_DEFAULT`) — both tiers receive the exact same
`effective_prompt` string, which already includes the
`last_planner_reasoning_note` enrichment added for Model 3. This tier
automatically benefited from that fix with zero additional code needed
— a positive confirmation that the fix was placed at the correct level
of abstraction (the shared `_synthesize_dpo_prompt` helper) rather than
duplicated per-tier, avoiding the exact "fixed one tier, forgot the
other" risk that duplication would have created.

## Connection to Model 3 (Polish-Default): `locked_regions` threading verified clean

Traced `swap_orchestrator.py::request_finalize()` → `_finalize_locked()`
→ `self.backends[active_tier].run(**run_kwargs)`: a clean `**kwargs`
pass-through the whole way. `locked_regions` is declared in
`QwenImageEditBackend.run()`'s signature and binds correctly;
`ZImageTurboBackend.run()` (Polish-Default) doesn't declare it and it's
silently absorbed into that backend's own `**kwargs` — confirmed this
is the correct, intentional behavior (only the Quality/edit tier does
per-region locking, matching PRD §5.4's framing), not a bug.

## Systematic dead-parameter check (per the pattern established in docs/review/32, 34, 35)

`run()`'s full parameter list — `handoff_image_ref`, `constraints`,
`locked_regions`, `prompt` — all confirmed actually read in the method
body. No dead parameters found for this backend.

## Test coverage

11 new tests in `test_polish_quality_backend.py` (previously zero direct
coverage for this backend). Full inference suite: 228 passed (up from
217). Full repo suite: 564 passed, 1 skipped, 0 failures.
