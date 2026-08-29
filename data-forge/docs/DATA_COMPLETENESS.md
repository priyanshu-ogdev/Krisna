# Data Completeness: PRD Model → Pipeline Output Trace

This document exists because "the model stack is finalized" and "the data
pipeline can train all of them" turned out to be two different claims —
the first was verified (licenses, VRAM budgets, weight availability) well
before the second was ever actually traced end-to-end. This is that trace,
kept as a standing reference rather than a one-time audit note, since it's
the answer to "does this model have real, usable data" for every
component in the PRD's final stack.

## Status by component

| PRD Component | Training Method | Data It Needs | Pipeline Stage(s) Producing It | Status |
|---|---|---|---|---|
| Planner (Qwen3.5-9B) | BF16 LoRA | Multi-turn conversation + tool-call JSON deltas | `s01_6_planner_synthesis` | **Fixed this revision** — was zero data path at all |
| Sketch tier (MaskGIT-lineage) | From scratch | UI-domain images + VQ tokens, `ui_first` only | `s08_encoding` Branch 3 (now domain-gated) | **Fixed this revision** — was running on every domain, wasting compute on data the model will never use |
| Z-Image-Turbo (polish, default) | QLoRA | Latents + captions, all domains | `s08_encoding` Branch 1 | Ready since the prior revision |
| Qwen-Image-Edit-2511 (polish, quality) | LoRA | Paired (source, instruction, target) edit examples | `s08_encoding` Branch 2 (backbone latents) + `s07_5_edit_pairs` (actual edit-task pairs) | **Fixed this revision** — backbone latents alone were a proxy, not the real task shape |
| Gemma 4 31B (Critic) | QLoRA | Real human-labeled critique/rating ground truth | `s01_5_uicrit_join` | **Fixed this revision** — was self-distillation only (Gemma judging its own output), now has real UICrit-sourced calibration data |

## What was broken, and why it stayed hidden

Three separate datasets (PD12M/CC12M, UICrit, Screen2Words) silently
ingested **zero usable records** before this revision, each for a
structurally similar reason: the generic fetch path assumes "this source
is a folder of standalone images with common extensions," and none of
these three actually are:

- **PD12M/CC12M** are metadata-only — a parquet file with an image-URL
  column, not bundled image files. Fixed via `fetcher.py`'s `url_list`
  download mode.
- **UICrit** isn't an image dataset at all — its screenshots are RICO's,
  and its actual value is ~983 human critique/rating annotations meant to
  be *joined* against RICO's already-ingested images. Fixed via the new
  `annotation_only` dataset flag and `s01_5_uicrit_join`.
- **Screen2Words** is captions *for* RICO's screens, the same join shape
  as UICrit, just for a different field (caption text instead of
  critique/rating). Fixed via the new `caption_join` download mode, which
  auto-detects between two possible real shapes (embedded images vs.
  ID-based join) since the live schema wasn't independently confirmed
  before this fix — see `fetcher.py::_fetch_huggingface_caption_join`'s
  docstring.

None of these were caught by a test before now, because no test checked
"does this dataset actually produce records" — see
`tests/test_dataset_ingestion_completeness.py`, added this revision
specifically to close that blind spot going forward. If a fourth dataset
turns out to have the same problem, that test file is where the pattern
should be checked first.

## Traceable data flow, end to end

```
UICrit repo (cloned, annotation_only)
        │
        ▼  s01_5_uicrit_join (filename-stem match against RICO)
RICO records gain critique_output.critique_source == "uicrit_human"
        │
        ├──────────────────────────────────────────────┐
        ▼                                                ▼
s01_6_planner_synthesis                          s12_model_data_export
(Tier-1 generates conversations,                 (critic_gemma4_31b/human_calibration.jsonl —
 seeded from real critique text,                  the REAL Gemma 4 QLoRA signal, kept
 validated against DesignStateDelta schema)        distinct from self_generated.jsonl)
        │
        ▼
planner_data/conversations/*.jsonl
        │
        ▼
model_data/planner_qwen3.5_9b/conversations/
```

```
Curated ui_first image (post-routing)
        │
        ▼  s07_5_edit_pairs (synthetic degradation: blur + downsample/upsample)
processed/edit_pairs/synthetic_ui/{id}_source.png (degraded)
                                    {id}_target.png (original)
                                    {id}.json (instruction)
        │                                    ▲
        │            processed/edit_pairs/external/{magicbrush,instructpix2pix}/
        │            (general-domain grounding, fetched directly by
        │             fetcher.py::_fetch_huggingface_triples)
        ▼
model_data/polish_qwenimage_edit_2511/edit_pairs/
```

## Fixed in this pass (post-v10 completeness review)

Two more bugs were found by tracing actual data flow end-to-end again
after the fixes above — neither was a missing data path (the class of
bug this document was originally written to track), but both would have
silently undermined the fixes already made:

- **Pre-flight storage projection ignored `sample_size` caps.**
  `s00_manifest_planning.py` summed every dataset's raw
  `expected_record_count` (PD12M's full 12.4M, CC12M's full 12.4M, etc.)
  instead of the actual capped/effective count that gets downloaded.
  That projected ~26M records / ~33TB against the PRD's real ~100K-500K
  / ~3TB target (§8.3), which would false-fail `pre_flight_check` on any
  normal workstation disk before Stage 1 ever ran. Fixed via
  `DatasetSpec.storage_relevant_record_count()` in `config.py`, which
  respects `fetch_config.sample_size` and zeroes out sources
  (`annotation_only`, `text_reference`, `caption_join`) that never
  produce a standalone image record. See `tests/test_storage_sizing.py`.
- **Gemma-4 critique generation could silently overwrite real human
  ground truth.** `s10_5_critic_preference.py` sampled from every
  audited/training_pool record with no exclusion for records that
  already carried `critique_output.critique_source == "uicrit_human"`
  from `s01_5_uicrit_join`. Since `Manifest.update_record()` overwrites
  `critique_output` wholesale (no merge), a RICO/UICrit record randomly
  selected into that stage's ~10% sample would have its human label
  silently replaced with a self-generated one — precisely the
  "self-distillation, not calibration" failure mode `s01_5_uicrit_join`
  exists to fix, undone by a downstream stage with no test catching it.
  Fixed via `_is_eligible_for_critic_sampling()`, which excludes
  already-human-labeled records from the sampling pool. See
  `tests/test_critic_preference_sampling.py`.

Also cleaned up in this pass, lower severity: `domain_tagger.py` was
missing explicit `pd12m`/`cc12m` entries in either source-key set (they
worked correctly only via the structural heuristic's default branch,
which a false-positive UI-element detection could have overridden), and
`pipeline.yaml` carried a stale "verified for Qwen-Image-2.0's VAE"
comment/dead-config block left over from before the Qwen-Image-2.0 ->
Qwen-Image-Edit-2511 swap — corrected/removed since Qwen-Image-2.0's
weights were never released and nothing about it could have been
verified.

## What's still explicitly out of scope

- **`/preference_pairs/`** — ranked A/B DPO pairs. Constructing genuine
  preference pairs needs multiple candidate generations from the
  product's own trained models to rank against each other; data-forge
  operates on found/curated images and generated singletons, not
  model-sampled candidates. This was true before this revision and
  remains true — see `s10_5_critic_preference.py`'s module docstring.
- **Live schema verification** for every newly-added dataset in this
  revision (Glaive, xLAM, MagicBrush, InstructPix2Pix, and the
  auto-detected paths for UICrit/Screen2Words) — column-name assumptions
  are documented as unverified at each site, with defensive detection and
  loud logging rather than silent failure, but "detects gracefully" is
  not the same claim as "confirmed against the live repo." Confirm before
  a production run, same discipline as every other unverified item
  flagged throughout this pipeline.
