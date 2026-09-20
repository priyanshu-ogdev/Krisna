# Phase 15 — Data-Forge Sources, Synthetic-Caption Sync, and Training-Sync Finalization

Scope for this pass, as asked: verify every model's training data comes
from proper, real sources; verify preprocessing; verify synthetic
(VLM-generated) caption handling is sound and actually reaches training
correctly; close out anything still out of sync between data-forge and
the training code that consumes its output.

Method unchanged from every prior phase: read the real code and configs,
don't take a docstring's word for its own correctness without checking
the file it claims to describe.

---

## §1 — Dataset sourcing: verified real, verified licensed, verified checked (not assumed)

Read `configs/datasets.yaml` in full. Every entry already carries the
discipline this review has been enforcing elsewhere: a real `repo_id`
or `repo_url`, a `license_url` and `license_notes` explaining exactly
what was independently confirmed vs. what still needs verification, and
`license_status: unverified` on every single entry (even ones with a
clean, confirmed license) — meaning the pipeline's own License
Verification Agent still runs on all of them rather than anything being
hardcoded as pre-cleared.

**Concrete evidence this is real diligence, not boilerplate:**
- RICO's own two ingestion paths (`rico_core`, `rico_semantic`) both
  carry inline comments documenting a **found-and-fixed bug**: both
  originally used `file_patterns` that matched zero files against the
  real repos (which ship parquet, not loose image files) — meaning
  *the single most foundational dataset in the UI-domain corpus silently
  fetched 0 records on every prior run*, and everything joined onto it
  (CLAY, Enrico, Screen2Words) would have been built on nothing. Fixed,
  with the real column names (`screenshot` vs. `image` — confirmed
  different between the two RICO mirrors, not assumed identical).
- CLAY's license was corrected from "unclear" to a confirmed CC BY 4.0
  citation directly from the paper's copyright footer — with the
  narrower remaining gap (no standalone LICENSE file in the specific
  repo used) stated explicitly rather than smoothed over.
- Two DPO preference datasets (`designsense_10k`, `designpref`) are
  correctly marked `repo_id: null` because they are **confirmed not
  publicly released yet** — not a placeholder for "haven't looked," a
  positive finding from checking the papers' own text and a
  cross-referencing survey paper (TASTE) that states outright
  DesignPref's data "has not yet been released." These are tracked via
  `s11_registry_watcher`'s scan targets rather than silently sitting
  broken.
- `uicrit`'s CC BY-ND license is flagged as a real legal tension with
  this pipeline's own recaption/re-annotation step (NoDerivs restricts
  redistributing modified versions) — surfaced as a flag for manual
  legal review, not resolved by assumption either way.

**Finding: none.** This is the one area of the whole project where I
did not find a gap — the dataset registry is honest about what's
verified, what's assumed, and what's outright missing, on every entry.

## §2 — Preprocessing: dedup, safety, quality — spot-checked for the same failure class already found elsewhere

Given `maskgit_vq` and the DPO-latents finding (§3 below) were both
"stage runs, silently produces something nothing downstream reads or
correctly gates on," checked the preprocessing stages for the same
pattern: does each stage's output actually get read by the stage after
it, with matching status strings?

- `s02_dedup` → `s03_quality`: confirmed `s03_quality` filters on the
  status `s02_dedup` actually sets (`"unique"`) — no drift.
- `s04_safety` → `s05_recaption`: confirmed `s05_recaption` filters on
  `r.status == "safety_classified" and r.safety_tier == "safe"`, which
  matches what `s04_safety` sets — no drift.
- `s05_recaption` → `s06_structure` → `s05_ocr_enrichment`: this one
  already had a **documented, fixed** bug in the code itself — the OCR
  sub-stage's filter used to check for `"recaptioned"` status, but by
  the time it runs, Stage 6 has already advanced records to
  `"structured"` within the same Tier-1 session, so the filter matched
  nothing and OCR silently never ran on any record, ever. Already fixed
  (comment in the code documents it) — re-confirmed the fix is
  consistent with the real stage order, not just present.

**Finding: none new.** The one real drift bug in this chain was already
caught and fixed in an earlier pass; re-verified rather than re-derived.

## §3 — Synthetic captions: the mechanism is sound and genuinely wired end-to-end (re-verified, not re-assumed)

This is the part of the ask I gave the most scrutiny, since "synthetic
data properly handled" is exactly the failure mode this project has hit
before (Phase 8's caption-style-mixing fix, and the general pattern of
docs claiming a fix that turned out not to be live in the actual
working copy — see the Critic-headroom regression from two sessions
ago). Traced the full chain fresh, file by file, rather than trusting
`07_consolidated_citations.md`'s summary of Phase 8:

1. `s05_recaption.py` generates a dense VLM caption per image via
   `Tier1Engine.generate_caption()`, passing the source dataset's own
   original caption in as `source_caption_hint` — explicitly a *hint*
   the VLM is told to "verify and expand on... don't just repeat it
   verbatim," not a ground truth to copy. Confirmed this hint actually
   reaches the prompt (string concatenation in `tier1.py`, not just
   accepted as a dead parameter).
2. `manifest.py` has a real `source_caption` column, and
   `s12_model_data_export.py` exports it into `captions.jsonl` alongside
   the dense `caption` — confirmed present in the actual export code
   (line 172), not just described in a comment.
3. `sync_sketch_tier.py` carries `source_caption` through into the
   Sketch tier's own `manifest.jsonl` — confirmed at three separate
   points in that file (read, default-to-None fallback, and the output
   dict), not just one.
4. `training/sketch/dataset.py`'s `SketchTokenDataset` does the actual
   95%-dense/5%-original mixing, **re-randomized per access** — matches
   Betker et al. (2023)'s own reported optimum ratio and randomization
   scheme, confirmed as a genuine per-`__getitem__` random draw
   (`random.random() >= self.caption_mix_ratio`), not a fixed per-image
   assignment that would defeat the point.
5. `train.py`'s `make_collate_fn` applies classifier-free-guidance
   conditioning dropout (`cfg_dropout_prob=0.1`, Ho & Salimans 2022) on
   top of whichever caption (dense or original) was selected — confirmed
   wired into the actual collate function used by the real `DataLoader`
   construction in `train()`, not a standalone helper never called.

**All five links confirmed live in the current working copy.** This is
the one synthetic-data mechanism in the project that, on this pass,
checked out completely — a genuine, well-implemented answer to "how is
VLM-generated caption data kept from silently mismatching what a real
user types at inference time."

## §4 — New finding: DPO latent-encoding stage is a second, still-live instance of the `maskgit_vq` orphaned-producer pattern

`s08_5_dpo_encoding.py` encodes preference pairs into Z-Image-Turbo's
latent space, intended for `train_dpo.py` to consume. **It doesn't.**
Confirmed by reading `train_dpo.py` directly: it resolves
`chosen_ref`/`rejected_ref` through the shared `BlobStore` and calls
`vae.encode(batch["chosen_pixel_values"]...)` itself, at train time,
from raw images — never touching `s08_5_dpo_encoding.py`'s
`.safetensors` output.

This was **already found and documented** in
`docs/review/02_polish_tier.md` (explicitly compared to the `maskgit_vq`
pattern there) — but, unlike `maskgit_vq`, it had not yet been acted on:
`configs/pipeline.yaml` still had `s08_5_dpo_encoding.enabled: true`,
meaning every full pipeline run was still spending real GPU-encode time
and disk space producing an artifact confirmed to have zero consumers.

**Fix applied this pass:** `enabled: false` in `pipeline.yaml`, with the
module docstring updated to explain why and what would need to be true
to re-enable it. Deliberately **not deleted outright** the way
`maskgit_vq` was — the two cases differ in one important way:
`maskgit_vq`'s encoder was *broken* (always raised `RuntimeError`), so
there was no real option other than removal. `s08_5_dpo_encoding.py`'s
encoder is presumed to work correctly (not independently verified in
this pass — no GPU available), it's just unconsumed. That leaves a real,
live choice — wire it into `train_dpo.py` as a fast path (skip live
VAE-encoding when precomputed latents exist, a genuine speed win if DPO
training ever becomes GPU-time-constrained) vs. delete it outright — and
manufacturing that decision under this review's own time constraints,
without being able to test either path against a real training run,
would be worse than disabling it and stating the tradeoff plainly for
whoever runs the next real DPO pass to decide.

**Verification:** `pipeline.yaml` parses correctly with the change;
confirmed no test in `tests/data_forge/` asserts this stage is enabled
(the one reference, in `test_preference_pairs_safety.py`, only concerns
a different stage's dedup-status guard). Full suite: 240 passed, same 5
pre-existing unrelated `pyarrow` failures.

## §5 — Cross-check: does every model actually get real data from a real place?

| Model tier | Real training data source | Verified path from data-forge to training |
|---|---|---|
| Sketch | RICO/CLAY/Enrico/Screen2Words/WebUI images + VLM-dense captions + original captions (mixed) | `sync_sketch_tier.py` → `SketchTokenDataset` — confirmed, §3 above |
| Polish-Default (dreambooth base) | data-forge's scrubbed/deduped/quality-gated `images/` + `captions.jsonl` | `sync_polish_default.py` — straight format bridge, correctly documented as not re-encoding (§ traced this pass, no issue found) |
| Polish-Default (DPO refinement) | Pick-a-Pic v2, HPDv2 (general aesthetic) + DesignSense-10k/DesignPref (UI-domain, when released) preference pairs | `sync_dpo_pairs.py` → `PreferenceStore` → `train_dpo.py` reads via `BlobStore`, re-encodes itself — confirmed, §4 above (this is the correct, live path; `s08_5_dpo_encoding.py` is the disabled, unconsumed one) |
| Planner | Frozen, RAG over UICrit (no fine-tuning) | Out of scope for this data-forge pass — UICrit's join (`s01_5_uicrit_join.py`) feeds the RAG corpus, not a training set; not re-audited here beyond confirming it's real, licensed data (CC BY-ND, flagged for legal review, §1) |
| Polish-Quality | Frozen, zero-shot ICL + SDEdit (no fine-tuning) | No training data needed — confirmed no data-forge stage claims to feed it |
| Critic | Frozen (Gemma 4, pre-quantized checkpoint) | No training data needed — confirmed no data-forge stage claims to feed it |

Every tier that trains on data-forge output now has a real, traced,
verified-consistent path from source dataset to training loop. The two
tiers with no training data need (Polish-Quality, Critic) correctly have
no data-forge stage claiming to feed them.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| Info | Dataset sourcing (`datasets.yaml`) — every entry real, licensed-or-flagged, independently checked | No action needed — this is the strongest-audited part of the whole project |
| Info | Preprocessing stage-to-stage status handoffs (dedup→quality, safety→recaption→OCR) | No action needed — the one real drift bug in this chain was already fixed in an earlier pass |
| Info | Synthetic caption generation → export → sketch-tier mixing → CFG dropout | No action needed — traced fresh end-to-end, all five links confirmed live |
| **Real, fixed** | `s08_5_dpo_encoding.py` was still `enabled: true` despite being confirmed (in an earlier phase's doc) to have zero real consumers — genuine wasted GPU/disk on every run | **Fixed**: disabled by default, decision (wire vs. delete) left explicit and undecided rather than silently made either way |

## Still open
- The wire-vs-delete decision for `s08_5_dpo_encoding.py` itself (§4) —
  intentionally left open, not resolved by this pass.
- Phase 1's RICO join/count verification — still the longest-standing
  open item across every phase of this review.
- Real hardware measurements for Polish-Default's VRAM (Phase 13) and a
  live rerun of `test_schema_validator.py` to confirm the stale-cache
  hypothesis from the prior session's summary.
