# Phase 23 — Closing the `inference/README.md` and `s08_5_dpo_encoding` Stale-Documentation Sweeps

Phase 22 found `inference/README.md` actively describing pre-fix VRAM
figures as current (Polish-Default at `8.0GB`, Critic at `~12GB/~40GB`,
`KRISNA_CRITIC_MAX_GPU_GB`'s documented default at `12.0` instead of
`11.5`) and flagged it, along with a same-class finding in `data-forge`'s
docs, as needing a fix. This phase confirms both fixes landed and
completes the second one, which had been left as "a known cleanup item"
under time pressure.

## §1 — `inference/README.md`: confirmed fixed

Verified directly against the file: the VRAM/RAM table now shows Polish
Default at `14.0GB / 0GB RAM` (full) / `12.0GB / 2.0GB RAM` (low-VRAM)
and Critic at `18.0GB / 0GB RAM` (full) / `11.5GB / 45.0GB RAM`
(low-VRAM, with an explicit "computed, ~0.5GB headroom" note), both
matching `model_registry.py`'s real current values exactly. The three
prose mentions of the old ~40GB figure, and the `KRISNA_CRITIC_MAX_GPU_GB`
documented default, are all corrected too, with an explicit "this used
to say X" note explaining why the old values were wrong — consistent
with this review's established correction style rather than a silent
rewrite. No further action needed here.

## §2 — `s08_5_dpo_encoding` stale-doc sweep: completed

Phase 22 found `data-forge/README.md` and `docs/data-forge/{ARCHITECTURE,DATA_COMPLETENESS,DATA_SOURCES}.md` all
described `s08_5_dpo_encoding` as an active, load-bearing stage that
`train_dpo.py` depends on — when Phase 15 had already found and
disabled it (`configs/pipeline.yaml`: `enabled: false`) after confirming
it has zero real consumers. That prior pass fixed the orchestrator code
comment and part of one doc, then flagged the rest as "a known cleanup
item" rather than finishing under time pressure. This phase finishes it.

## What was actually wrong, file by file

Checked each file's *specific* claim against the real code before
editing — not a global find-replace:

- **`docs/data-forge/DATA_COMPLETENESS.md`** — the training-readiness
  status table routed the DPO row through `s01_6_preference_pairs` →
  `s08_5_dpo_encoding` as the pipeline source, and a separate flow
  diagram showed `s08_5_dpo_encoding → s12_model_data_export →
  model_data/dpo_alignment/general/...` as the real data path. Neither
  is true: `train_dpo.py` never reads `model_data/dpo_alignment/` or
  any `s08_5_dpo_encoding.py` output — it resolves pairs through
  `training/data_forge_bridge/sync_dpo_pairs.py`, which reads Stage
  1.6's raw images directly via the shared `BlobStore` into a
  `PreferenceStore` sqlite db, and live-encodes them at train time.
  Fixed the table row and redrew the flow diagram to show the real path.
- **`docs/data-forge/ARCHITECTURE.md`** — the Stage 8.5 table row, a
  "DPO alignment artifacts" section, and two further prose mentions all
  described the stage as functioning normally. Fixed all four; left the
  one changelog line ("Added `s01_6_preference_pairs.py` and
  `s08_5_dpo_encoding.py` for real, human-labeled DPO data") alone,
  since as a historical record of what was added when, it's accurate —
  the stage *was* added, it just isn't what actually runs the DPO data
  path today.
- **`docs/data-forge/DATA_SOURCES.md`** — Pick-a-Pic v2's row cited
  `s08_5_dpo_encoding.py` as part of its consumption path. Fixed to cite
  the real path (`sync_dpo_pairs.py` → `train_dpo.py`).
- **`data-forge/README.md`** — the stage table, a "two independent
  streams" architecture explainer, a `--stages 8.5` example command's
  surrounding prose, and a second stage-table appearance all described
  the same false active-stage claim. Fixed all four, with an explicit
  "this used to say X" correction in the architecture explainer (the
  most-read section) rather than a silent rewrite, matching the
  correction style already used for the `model_registry.py` comment fix
  in Phase 21.

## Verification

Full-text swept every remaining `s08_5`/`Stage 8.5`/`8.5` mention across
`docs/data-forge/*.md` and `data-forge/README.md` after editing:
every remaining hit is either the correction text itself, or the one
changelog line correctly left alone as historical record — none present
the disabled stage as active. Full test suite (doc-only changes, run to
confirm no accidental syntax breakage in surrounding code blocks):
**356 passed, 10 skipped**, same as before this phase.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, confirmed fixed** | `inference/README.md`'s VRAM/RAM table and three prose mentions described pre-fix Polish-Default (8.0GB) and Critic (~12GB/~40GB) figures as current | Confirmed fixed — verified directly against the file, matches `model_registry.py` exactly |
| **Real, fixed (completes a prior pass)** | `s08_5_dpo_encoding` described as active/load-bearing across `data-forge/README.md` and three `docs/data-forge/*.md` files, despite being disabled since Phase 15 | Fixed — table rows, flow diagrams, and prose all corrected against the real `sync_dpo_pairs.py` → `train_dpo.py` path, with explicit "this used to say X" corrections rather than silent rewrites |

## Still open, unchanged from Phase 22
- Phase 1's RICO join/count verification.
- Real hardware VRAM/RAM measurements for every tier.
- `s08_5_dpo_encoding.py`'s wire-vs-delete decision (Phase 16 recommends
  deletion; still not executed).
- `training/README.md` was checked in Phase 22's pass and found
  incomplete (missing mentions of newer additions like GameLabel-10K,
  gradient checkpointing) but not actively wrong — lower priority than
  the active-falsehood class this phase closed out, still worth a pass.
