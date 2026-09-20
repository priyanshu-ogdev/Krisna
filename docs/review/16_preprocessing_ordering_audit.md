# Phase 16 — Preprocessing Execution-Order Bug (Escalation-Rescued Records Silently Dropped)

Direct answer to being asked to double-check the "no mismatch in
preprocessing" claim from the previous pass: **that claim was wrong.**
This phase found a real one, structurally the same *impact* as the
`maskgit_vq` bug (silent, permanent loss of eligible training records)
but a different *cause* — not a broken encoder, an execution-order bug
in the orchestrator that no prior phase's pairwise stage-filter checks
could have caught, because it's not a filter mismatch between two
adjacent stages, it's a mismatch between the declared `requires` graph
and the actual runtime sequencing.

## What was wrong

`s04_safety` marks a record either `safety_tier="safe"`,
`safety_tier="unsafe"` (excluded immediately), or
`safety_tier="borderline"` (kept in the pipeline, flagged for a second
opinion). `s04_5_escalation` is the only stage that can resolve a
`"borderline"` record to `"safe"` — its whole purpose is rescuing
records a second, independent model call confirms are actually fine.

`s05_recaption` (and `s06_structure` after it) only process records
with `safety_tier == "safe"`. That's correct *filter* logic — the bug
was never in what any single stage checks, it's in **when** each stage
actually ran.

Read `orchestrator.py`'s real `execute_pipeline()` method directly
(not its own comments, which describe the *intended* design correctly —
the actual code had drifted from it) and confirmed: `s05_recaption` and
`s06_structure` were grouped into the same single vLLM session as
`s04_safety`, running **immediately after** it — while
`s04_5_escalation` was structurally placed in a **separate, later**
block, after the OCR sub-phase. So for any given chunk of records, the
real execution order was:

```
s03_quality → s03_5_pii_scrub → s04_safety → s05_recaption → s06_structure
  → s05_ocr_enrichment → s05_5_pii_text_redact → s04_5_escalation → s07_routing → s08_encoding
```

By the time escalation ran and flipped a rescued record's
`safety_tier` to `"safe"`, `s05_recaption` and `s06_structure` had
**already finished executing for that chunk** — and nothing in
`execute_pipeline()` loops back to re-run them. The rescued record's
`status` stays permanently at `"safety_classified"` (the value
`s04_safety` set for `"borderline"` records) — it never becomes
`"recaptioned"`, never `"structured"`, never `"routed"`, never
`"encoded"`, never reaches `training_pool` or `heldout`.

**Every record the two-tier escalation system successfully rescues was
being silently discarded — the exact opposite of what escalation exists
to do.** No error, no warning, no failed test — the record just quietly
stops advancing and is absent from the final training pool.

## Why prior phases didn't catch this

Every earlier phase's preprocessing review (including this same
session's own §2, one message ago) checked producer→consumer status
consistency **pairwise**: does stage N's output status match what
stage N+1 filters on? Every pairwise check in the chain was individually
correct — `s05_recaption`'s filter genuinely does match what
`s04_safety` sets. The bug only exists at the level of **global
execution order across more than two stages**, which no pairwise check
can see. Caught here specifically by reading `execute_pipeline()`'s
actual phase-by-phase control flow end to end, not by re-checking
individual stage files again.

## Fix applied

Restructured `execute_pipeline()`'s phase ordering:

- **Phase 2** (single vLLM session) now runs only `s03_quality`,
  `s03_5_pii_scrub`, `s04_safety` — `s05_recaption`/`s06_structure`
  removed from this session.
- **Phase 3** (moved earlier, was "Phase 4"): `s04_5_escalation` now
  runs immediately after Phase 2, resolving whatever it can.
- **Phase 4** (new): `s05_recaption` + `s06_structure` run in their own
  vLLM session, **after** escalation — `record_ids` is freshly
  re-filtered at this point, so it correctly includes both records that
  were `"safe"` from the start and any escalation-rescued ones, since
  both now genuinely satisfy `safety_tier == "safe"`.
- **Phase 5** (was "Phase 3"): OCR + text-PII-redaction, unchanged in
  content, just renumbered — now correctly runs after structure for
  *all* eligible records, including rescued ones (it was also silently
  missing rescued records before this fix, for the identical reason).
- `s07_routing` and `s08_encoding` unchanged — they were always
  correctly positioned after everything above.

No stage's own filter logic changed — the fix is entirely in
`orchestrator.py`'s phase sequencing, matching what `docs/architecture/
SYNC_DESIGN.md` and the stages' own `requires` declarations already
implied was the intent (recaption depends on safety being fully
resolved, not just first-pass-classified).

## Verification

Added `TestEscalationRunsBeforeRecaptionAndStructure` to
`tests/data_forge/test_orchestrator.py` — not a unit test of one stage
in isolation, a test of `execute_pipeline()`'s actual call order.
Monkeypatches `_run_stage` to record invocation order (no real VLM
inference needed — this bug is about sequencing, not model output),
sets up a manifest record with `safety_tier="borderline"` (the exact
condition that exposes the bug — escalation only has work to do when
this exists), and asserts `s04_5_escalation` is called before both
`s05_recaption` and `s06_structure`, and after `s04_safety`.

**Test passes against the fixed code.** I attempted to also confirm it
fails against the original buggy order by reconstructing that order in
a scratch copy of the file; the reconstruction I tried was incomplete
(reverting one list without its paired condition) and didn't actually
reproduce the original bug, so I'm not reporting a clean before/after
differential here — stating that plainly rather than implying a
verification I didn't actually get clean. What I have high confidence
in instead: the original buggy order was confirmed by directly reading
the unmodified source (`sed` output captured and read line-by-line, not
inferred), and the fixed order was confirmed the same way after editing.

Full suite: **241 passed** (up from 240 — the new test), same 5
pre-existing unrelated `pyarrow` failures.

## Answering the three "still open" items directly, as asked

**`s08_5_dpo_encoding.py`, wire vs. delete:** my honest opinion —
delete it, don't wire it. The case for wiring (skip live VAE-encoding
in `train_dpo.py` when precomputed latents exist) only pays off if DPO
training becomes GPU-time-constrained, which isn't established as true
anywhere in this project's docs; it's speculative future-proofing for a
bottleneck that may never materialize, at the cost of a second code path
(precomputed-latent vs. live-encode) in the training loop that has to
stay correct forever after. `maskgit_vq` already established this
project's own precedent for exactly this situation: an orphaned
producer with no real consumer gets removed, not preserved on the
chance something uses it later. I didn't make this call unilaterally in
the last pass because I wanted to flag it rather than decide it
silently — but if asked for a recommendation, this is it.

**RICO join count:** still genuinely unverified — it requires running
the real pipeline against live data, which isn't possible in this
sandbox (no GPU, no network access to the actual HF/GitHub sources).
Nothing new to report; still the single most consequential remaining
unknown, since it determines whether the paper's headline corpus-size
claim is accurate.

**Real hardware VRAM measurements:** same — genuinely needs a GPU this
environment doesn't have. Nothing to add beyond the arithmetic-grounded
estimates already in `model_registry.py`.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, fixed — highest severity found in this whole review** | Escalation-rescued records were silently, permanently dropped from the training pool — `s04_5_escalation` ran after, not before, the `safety_tier=="safe"` filters in `s05_recaption`/`s06_structure` | **Fixed**: `execute_pipeline()`'s phase order corrected; regression test added and passing against the fixed code |
| Info | Three previously-open items reviewed for an opinion as asked | Recommend deleting `s08_5_dpo_encoding.py` rather than wiring it; RICO count and VRAM measurements remain genuinely blocked on hardware/network access this environment doesn't have |
