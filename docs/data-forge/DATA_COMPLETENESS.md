# Data Completeness: PRD Model → Pipeline Output Trace

This document traces, for every PRD model component, whether data-forge
actually produces what its training method needs — not just whether the
model stack itself is finalized. It's kept as a living reference, not a
one-time audit note.

**v16 update (no-RLHF-loop revision):** three of the five components
below (Planner, Qwen-Image-Edit-2511, Gemma-4 Critic) now ship **frozen**
— they were removed from data-forge's training scope entirely, not
"fixed" by adding more data. The status table and data-flow diagrams
immediately below reflect the *current* state. Everything under
"Historical record" further down describes the pipeline as it existed
before this restructuring — kept as the reasoning trail behind several
still-current decisions (e.g. why UICrit is ingested via a dedicated join
rather than the generic scanner), not because it describes current
behavior. Stages it references (`s01_6_planner_synthesis`,
`s07_5_edit_pairs`, `s10_5_critic_preference`) no longer exist in this
codebase.

## Status by component (current, v16)

| PRD Component | Training Method | Data It Needs | Pipeline Source | Status |
|---|---|---|---|---|
| Planner (Qwen3.5-9B) | **None — frozen** | Real critique text for RAG retrieval (not training data) | `s01_5_uicrit_join` → `s12`'s `planner_rag_corpus/` | **Complete.** UICrit's real human critiques, retrieved at inference, no fine-tune. |
| Sketch tier (MaskGIT-lineage) | From scratch | UI-domain images + VQ tokens, `ui_first` only | `s08_encoding` (domain-gated) | **Complete.** ~410K real UI images across RICO/CLAY/Enrico/WebUI (see `DATA_SOURCES.md` for the count breakdown). |
| Z-Image-Turbo (polish) | LoRA/QLoRA | Latents + captions, all domains | `s08_encoding` | **Complete.** PD12M/CC12M + the UI pool above. |
| Z-Image-Turbo DPO alignment | Diffusion-DPO | Real human preference pairs | `s01_6_preference_pairs` → `train_dpo.py` (live-encodes; see note below) | **Partially complete.** Stage-1 (general aesthetic: Pick-a-Pic v2 + HPDv2) is real and fetchable. Stage-2 (UI/design-domain: DesignSense-10k + DesignPref) has **zero currently-fetchable pairs** — both datasets are real and published but neither has a confirmed public download location as of this revision; see `DATA_SOURCES.md`. |
| Qwen-Image-Edit-2511 (polish, quality) | **None — frozen** | none | — | **Complete by design.** Zero-shot ICL + SDEdit-style inference-time conditioning; no paired edit-task data needed at all. |
| Gemma 4 31B (Critic) | **None — frozen** | none | — | **Complete by design.** Zero-shot VLM-as-judge, on-demand product feature; data-forge never loads this model. |

**Correction (this revision):** the table above used to route the DPO
column through `s08_5_dpo_encoding` as if it were the live encoding
step. It isn't — that stage is **disabled by default**
(`configs/pipeline.yaml`: `s08_5_dpo_encoding.enabled: false`), confirmed
to have zero real consumers: `training/polish/train_dpo.py` resolves
`chosen_ref`/`rejected_ref` through the shared `BlobStore` and calls
`vae.encode(...)` on the raw images itself, at train time, never reading
`s08_5_dpo_encoding.py`'s `.safetensors` output. This is the same
bypass pattern as `sync_sketch_tier.py` re-tokenizing Sketch-tier images
instead of trusting data-forge's own encoding — see
`s08_5_dpo_encoding.py`'s own module docstring and
`docs/review/15_data_forge_finalization.md` for the full history. What
data-forge is actually responsible for, and does correctly, is
`s01_6_preference_pairs` — deduping, safety-scrubbing, and staging the
raw preference-pair images and prompts that `train_dpo.py` then reads
directly.

The one open item that actually matters for training readiness: **Stage-2 UI-domain DPO data doesn't exist publicly yet.** This isn't a pipeline defect — `s01_6_preference_pairs` is built and tested and will pick the data up automatically the moment DesignSense-10k or DesignPref gets a real public repo (the registry watcher, Stage 11, is configured to flag exactly this). Until then, Z-Image-Turbo's DPO alignment runs on Stage-1 (general aesthetic) data only — a real, disclosed limitation to report as such, not paper over, per the PRD's own ablation (c): does general-preference DPO transfer to the UI domain, evaluated against TASTE's held-out multi-axis human agreement.

## Traceable data flow, end to end (current, v16)

```
UICrit repo (cloned, annotation_only)
        │
        ▼  s01_5_uicrit_join (filename-stem match against RICO)
RICO records gain critique_output.critique_source == "uicrit_human"
        │
        ▼  s12_model_data_export
model_data/planner_rag_corpus/uicrit_critiques.jsonl
        │
        ▼  (retrieved at Planner INFERENCE time — never fine-tuned on;
        │   the Planner ships frozen, so this is the entire "training
        │   data" story for it: a retrieval corpus, not a training set)
```

```
Pick-a-Pic v2 / HPDv2 (real human preference comparisons)
        │
        ▼  s01_fetch (preference_pair / hpdv2_ranked_list adapters)
preference_pairs/{pickapic_v2,hpdv2}/{pair_id}.json + _a.png/_b.png
        │
        ▼  s01_6_preference_pairs (cross-source dedup, face-blur,
        │   NSFW/safety classification — drops unsafe pairs entirely)
        │
        ▼  training/data_forge_bridge/sync_dpo_pairs.py (NOT s08_5_dpo_
        │   encoding + s12_model_data_export — see the correction above.
        │   Resolves each pair's raw images through the shared BlobStore
        │   into a PreferenceStore sqlite db)
        ▼  train_dpo.py reads the db directly, live-encoding each pair
           through Z-Image-Turbo's own VAE at train time (same latent
           space the policy model generates in — same reasoning
           s08_5_dpo_encoding.py's now-unused code intended, just
           executed at train time instead of pre-computed)
```

DesignSense-10k and DesignPref would flow through the identical path
once fetchable — the code is built and tested end-to-end for this, but
as of this revision both datasets have
`repo_id: null` in `datasets.yaml` (see the status table above and
`DATA_SOURCES.md`), so this path currently produces zero output for
either. `s01_fetch` logs a clean `missing_repo_id` error rather than
silently skipping with no signal.

## What's still explicitly out of scope

- **Stage-2 UI-domain DPO data (DesignSense-10k, DesignPref)** — real,
  published datasets, but neither has a confirmed public download
  location as of this revision. Not a pipeline gap; a data-availability
  gap outside this pipeline's control. Re-check via the registry watcher
  (Stage 11) periodically.
- **Live schema verification** for auto-detected column/join-key
  assumptions across every fetch adapter — documented as unverified at
  each site, with defensive detection and loud logging rather than
  silent failure, but "detects gracefully" is not the same claim as
  "confirmed against the live repo." Confirm before a production-scale
  run, same discipline as every other unverified item flagged throughout
  this pipeline.
- **Model-revision pinning — RESOLVED.** Every `revision: "main"` in
  `models.yaml`/`datasets.yaml` was unpinned by design (this repo was
  built without live network access to resolve real commit SHAs).
  **`scripts/data-forge/pin_revisions.py` now exists** — it was
  referenced throughout this codebase's comments for a long time as the
  fix but never actually written; it's real now, not just documented.
  Verified: correctly resolves every real pinnable entry (6 in
  `models.yaml`, all `source_type: "huggingface"` entries in
  `datasets.yaml` with a non-null `repo_id`) via the HuggingFace API,
  skips already-pinned entries and `repo_id: null` entries (DesignSense-
  10k/DesignPref) without erroring, and — the part that would have
  shipped a real bug if untested — produces a byte-for-byte no-op
  round-trip on everything it doesn't touch and a single-line, comment-
  preserving diff on what it does (an earlier draft of this script
  silently reformatted every list in both files and turned `null` values
  into empty scalars on every run, a real `ruamel.yaml` default-settings
  quirk, caught by literally diffing a load-then-dump round trip against
  the untouched file before trusting it). See
  `../../tests/data_forge/test_pin_revisions.py`.

  ```bash
  python scripts/data-forge/pin_revisions.py           # dry run, prints only
  python scripts/data-forge/pin_revisions.py --apply    # writes real SHAs
  ```
  manually.

---

## Historical record: the pre-v16 investigation (superseded, kept as reasoning trail)

*Everything below this line describes the pipeline as it existed before
the no-RLHF-loop restructuring. `s01_6_planner_synthesis`,
`s07_5_edit_pairs`, and `s10_5_critic_preference` — referenced throughout
this section — no longer exist in this codebase.*

### What was broken, and why it stayed hidden (pre-v16)

Three separate datasets (PD12M/CC12M, UICrit, Screen2Words) silently
ingested **zero usable records** at one point, each for a structurally
similar reason: the generic fetch path assumes "this source is a folder
of standalone images with common extensions," and none of these three
actually are:

- **PD12M/CC12M** are metadata-only — a parquet file with an image-URL
  column, not bundled image files. Fixed via `fetcher.py`'s `url_list`
  download mode.
- **UICrit** isn't an image dataset at all — its screenshots are RICO's,
  and its actual value is ~983 human critique/rating annotations meant to
  be *joined* against RICO's already-ingested images. Fixed via the
  `annotation_only` dataset flag and `s01_5_uicrit_join` (still active,
  now feeding the Planner's RAG corpus instead of a fine-tune).
- **Screen2Words** is captions *for* RICO's screens, the same join shape
  as UICrit, just for a different field. Fixed via the `caption_join`
  download mode.

None of these were caught by a test at the time, because no test checked
"does this dataset actually produce records" — see
`../../tests/data_forge/test_dataset_ingestion_completeness.py`, added specifically to
close that blind spot going forward.

Two further bugs were found by tracing data flow end-to-end a second
time, neither a missing data path but both capable of silently
undermining the fixes already made:

- **Pre-flight storage projection ignored `sample_size` caps** — summed
  every dataset's raw `expected_record_count` instead of the actual
  capped count, projecting ~26M records / ~33TB against the PRD's real
  ~100K-500K target. This exact class of bug recurred in v16 (see below)
  and is now fixed at the source via `DatasetSpec.storage_relevant_record_count()`.
- **Gemma-4 critique generation could silently overwrite real human
  ground truth** — `s10_5_critic_preference.py` sampled with no
  exclusion for records already carrying `critique_output.critique_source
  == "uicrit_human"`, and `Manifest.update_record()` overwrites
  `critique_output` wholesale. A RICO/UICrit record could have its human
  label silently replaced with a self-generated one. Moot now — that
  stage no longer exists.

### v16 restructuring: what was found and fixed during the no-RLHF-loop revision

- **`../../data-forge/data_forge/utils/completeness.py`** still required `qwen_image_latent` after
  Qwen-Image-Edit-2511's encoding branch was removed from Stage 8 —
  would have flagged every record in the corpus as incomplete, starving
  Stage 9/10/12 of all input. Fixed.
- **`storage.py`'s pre-flight check** had the exact `sample_size`-
  ignoring bug described above recur in a different form after the
  restructuring touched dataset counting logic again. Fixed via the same
  `storage_relevant_record_count()` mechanism, now also correctly
  zeroing out preference-pair and eval-only sources that don't produce
  standalone image records.
- **Preference-pair images had zero safety/NSFW classification** — an
  early pass of `s01_6_preference_pairs.py` ran with no engine at all.
  Fixed: now runs the same `classify_safety` path the main manifest
  uses, dropping any pair where either image is flagged unsafe.
- **HPDv2's real schema** (a variable-length ranked list, not a fixed
  two-image table) didn't match the generic `preference_pair` fetcher —
  confirmed directly against the live `ymhao/HPDv2` dataset card. Fixed
  with a dedicated `hpdv2_ranked_list` adapter.
- **RICO core/semantic** were configured to scan for loose image files
  against repos that ship exclusively parquet-embedded images — the same
  failure class as the pre-v16 PD12M/CC12M/UICrit/Screen2Words bugs,
  just missed in that earlier pass. This is the single most consequential
  bug found across every review of this pipeline: RICO is what CLAY,
  Enrico, and Screen2Words all join onto, so a silent zero-record RICO
  fetch would have collapsed the entire UI-domain training pool with
  nothing downstream erroring to explain why. Fixed with a dedicated
  `hf_parquet_images` fetch adapter, with regression tests. Both repos'
  image-column names confirmed directly against their live dataset cards
  and hardcoded (`"screenshot"` for `creative-graphic-design/Rico`,
  `"image"` for `Voxel51/rico`) rather than left to runtime
  auto-detection — the two repos independently export the same
  underlying screens under different column names.
- **`cli.py`'s `--stages` shortcut map** still pointed at deleted stage
  names after the removal. Fixed.
