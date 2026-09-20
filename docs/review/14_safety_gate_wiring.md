# Phase 14 — Safety Gate Wiring (VerifierStack.safety_gate() Was Defined But Never Called)

Note on this doc's own history: the fix described here was implemented
and tested earlier in this review's working session, and both
`07_consolidated_citations.md` and `docs/review/README.md` were already
updated to reference this file — but the file itself was never actually
written to disk, an oversight caught and corrected in a later pass while
verifying the review's own docs are honest about what's actually been
done (the same standard this review has applied to the codebase
throughout: a doc claiming a fix is not itself evidence the fix exists).
Confirmed before writing this: `orchestrator/flows.py`'s `_check_safety_gate()`
function and its call site inside `finalize()` are genuinely present in
the current code, and both new tests in `tests/inference/test_flows.py`
(`TestFinalizeFlowSafetyGate`) pass. The fix is real; only this
write-up was missing.

## What was wrong

`inference/src/krisna_inference/verifiers/verifier_stack.py` defines
`VerifierStack.safety_gate(image, min_safety_score=0.9) -> tuple[bool, float]`
— its own docstring describes it explicitly as a pass/fail **gate**,
something that should block a result before the user ever sees it, not
just another score sitting alongside the rest.

Grepped the entire `inference/` tree for callers of `safety_gate`:
none. `orchestrator/flows.py`'s `finalize()` calls
`_run_verifier_stack()`, which calls `verifier_stack.score_finalize_output(...)`
— the five quality/alignment scores (`clip_alignment`, `ocr_readability`,
`layout_iou`, `aesthetic`, `handoff_consistency`) — and nothing else.
`state.stage = SessionStage.FINALIZED` happens unconditionally right
after, regardless of what any score (had one existed) said about safety.

**A method whose entire purpose is documented as "block before the user
sees it" was dead code.** Every finalized image, including a genuinely
unsafe one, would finalize successfully with zero safety enforcement —
not a low score buried in an unused field, no enforcement of any kind.

## Fix applied

- `flows.finalize()` gained a `min_safety_score: float = 0.9` parameter
  (forwarded to `safety_gate()` unchanged).
- A new `_check_safety_gate()` helper loads the polished image from the
  blob store and calls `verifier_stack.safety_gate()`.
- Wired into `finalize()`: when a real `verifier_stack` is present and
  the output is a real blob ref, the gate runs **before**
  `score_finalize_output()` and before the stage is marked
  `FINALIZED`. On failure: `state.stage` rolls back to `SKETCHING`
  (mirroring the existing `not result.ok` rollback path exactly), and a
  `FlowError` is raised — already correctly caught by `service.py`'s
  `/finalize` route and turned into a clean HTTP error, no route changes
  needed.
- Confirmed this is reachable in the real service, not just in tests:
  `service.py` only constructs a real `_verifier_stack` (via
  `get_verifier_stack()`) when `KRISNA_USE_REAL_BACKENDS=1`; MockBackend
  mode correctly passes `verifier_stack=None`, since Mock doesn't
  produce real images to score in the first place.

## Verification

Added `TestFinalizeFlowSafetyGate` to `tests/inference/test_flows.py`:
two tests exercising the actual `finalize()` code path (not the gate
method in isolation) against a real `blob://` image ref — MockBackend's
own `render://...` refs never satisfy the `.startswith("blob://")`
check, so a real blob had to be constructed via the blob store to reach
this branch at all.

- `test_unsafe_image_blocks_finalize_and_rolls_back_stage` — a fake
  `VerifierStack.safety_gate()` returns `(False, 0.1)`; asserts
  `FlowError` is raised with "safety gate failed" in the message, the
  fake `score_finalize_output()` (which raises if called) never runs,
  and the reloaded session's stage is `SKETCHING`, not `FINALIZED`.
- `test_safe_image_passes_gate_and_scores_still_populate` — a fake gate
  returns `(True, 0.99)`; asserts the flow completes normally and
  `verifier_scores` are still populated from `score_finalize_output()`.

Both pass. Full `tests/inference` suite unaffected otherwise.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, fixed** | `VerifierStack.safety_gate()` was defined, documented as a hard gate, and never called anywhere — no safety enforcement on any finalized image | **Fixed**: wired into `finalize()`, blocks and rolls back on failure, verified against the real code path with a real blob ref, not just the gate method in isolation |
| Info (process, not code) | This doc itself was referenced by two other docs before it existed | Fixed by writing it now, after re-confirming the underlying code fix is genuinely live — not assumed from the citation alone |
