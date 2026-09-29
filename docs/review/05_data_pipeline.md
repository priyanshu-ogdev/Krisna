# Phase 5 — Data Pipeline (data-forge, cross-cutting)

> **RESOLVED (post-Phase-1 follow-up session):** this phase's completeness
> logic (`utils/completeness.py`, `s09_heldout.py`,
> `s12_model_data_export.py`) has been corrected as a direct consequence
> of the Phase 1 `maskgit_vq` fix — see
> `docs/review/12_post_upgrade_resync_audit.md` §1. Re-verified with the
> full `tests/data_forge` suite (114 passed).

This phase closes out the licensing/dedup question left open in Phase 1,
and reviews the remaining cross-cutting correctness properties (safety,
PII, domain balance) that every model phase depends on.

## Resolved: RICO core/semantic license-exclusion → dedup interaction
**Closing the open item from Phase 1.** Traced stage order directly:
`s01_fetch` runs `LicenseVerificationAgent` and updates every `rico_core`
record to `excluded_pending_review` status at fetch time (dataset-scoped
via `query_by_status_and_dataset`). `s02_dedup.py` explicitly filters to
`r.status == "fetched"` at its very first step (line 33) before building
its dedup pool. **Confirmed: a license-excluded `rico_core` never enters
dedup**, so there's no risk of the CLIP-similarity near-duplicate pass
keeping the excluded copy over `rico_semantic`'s usable one. The UICrit
join stage's `_RICO_SOURCE_DATASETS = ("rico_core", "rico_semantic")`
already treats both as valid join targets, not just `rico_core` — so the
join mechanism itself doesn't assume `rico_core` survives. **Net: the
~410K usable-image figure's dependency chain is sound as designed**,
though the *exact* final count still depends on `rico_core` actually
being excluded (vs. eventually getting a manual license sign-off) and on
how many CLAY/Enrico/Screen2Words records successfully resolve against
`rico_semantic` specifically — worth computing that exact number from a
real pipeline run before citing "~410K" as a fixed figure in the paper,
since it's presented as a point estimate in `DATA_COMPLETENESS.md` but
depends on this three-way interaction.

## Safety / PII pipeline — real, staged correctly
Order confirmed via `pipeline.yaml`: `s01_fetch` (license) → `s02_dedup`
→ `s03_quality` → `s03_5_pii_scrub` (image PII) → `s04_safety` (NSFW/
safety classification) → `s04_5_escalation` → `s05_recaption` →
`s05_ocr_enrichment` → `s06_structure` → `s05_5_pii_text_redact` (text
PII, runs after OCR specifically so it has OCR'd text to redact) →
`s07_routing` (domain balance) → `s08_encoding`. This ordering is
correct on its own terms: PII scrubbing happens before safety
classification (so safety doesn't score on unredacted faces/PII), and
text-PII redaction is correctly deferred until *after* OCR populates the
text to redact in the first place — a real ordering dependency, not
arbitrary stage numbering. Also independently confirmed in Phase 2:
`s01_6_preference_pairs.py` was found to have originally skipped safety
classification for preference-pair images and was fixed to run the same
`classify_safety` path as the main manifest — verified as actually fixed
in this pass, not just claimed (`s01_6_preference_pairs.py` calls the
same engine-backed classifier, dropping any pair where either image is
flagged unsafe).

## Domain balance (`ui_first_ratio`) — known limitation, correctly documented, not re-litigated here
Already covered in `DATA_SOURCES.md` and cited in Phase 1: `ShardRouter.route()`
enforces the configured `ui_first_ratio` (default 0.70) per-chunk, not
cumulatively corpus-wide, so the realized ratio can drift slightly
across chunks with different domain compositions even though each chunk
is individually correct. This is a genuine, documented, low-severity
limitation — flagged in-repo as a real open item for a future revision,
not something this review needs to re-derive. Worth stating in the paper
as a known caveat on the actual training-set domain ratio, not treated
as a hard 70/30 guarantee.

## Model-revision pinning — verified resolved, not just claimed
`DATA_COMPLETENESS.md` claims `scripts/data-forge/pin_revisions.py` now
exists and correctly resolves every pinnable `models.yaml`/`datasets.yaml`
entry via the HuggingFace API, verified via a byte-for-byte round-trip
test against `ruamel.yaml` reformatting quirks. Confirmed the script and
its test (`tests/data_forge/test_pin_revisions.py`) both exist on disk in
this pass — did not re-run it against live HF API (no network access in
this environment), so the *mechanism* is verified present and tested,
but a live pin-and-verify run against current HF commit SHAs is still
the actual production-readiness gate before a real training run, exactly
as the repo's own docs already state.

## Findings summary

| Severity | Finding |
|---|---|
| Resolved | RICO core/semantic license-exclusion correctly can't corrupt dedup — verified via actual stage-order + status-filter tracing, not assumed |
| Action | Compute the *exact* final usable-image count from a real pipeline run before citing "~410K" as a fixed figure — it depends on a three-way interaction (license exclusion, dedup, cross-dataset join resolution) that's sound by design but not independently reproduced with real numbers in this review |
| Info | Safety/PII stage ordering is correct and dependency-aware (PII before safety scoring; text-PII redaction after OCR populates text) |
| Info | `ui_first_ratio` per-chunk (not corpus-wide) enforcement remains a known, documented, low-severity limitation — not re-flagged as new |
| Info | Revision-pinning script exists and is tested; still needs one live run against real HF API before production training, per the repo's own stated gate |

## Citations (Phase 5)
No new external citations for this phase — it's an internal pipeline-correctness review. Dataset citations already listed in Phase 1's findings (RICO/CLAY/Enrico/WebUI/Screen2Words/UICrit) apply here as the underlying sources this pipeline processes.

---
Next: Phase 6 — Inference orchestrator (SwapOrchestrator, VRAM budget, verifiers), the last phase in the plan, followed by Phase 7 (consolidated citations doc merging all findings).
