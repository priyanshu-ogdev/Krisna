# Synthetic Data Audit & Generalization Coverage

Scope: every dataset and label that reaches any trained model in this
repo (Sketch tier, Polish LoRA, Polish DPO), checked for whether it's
**real** (human-created images/text/preferences) or **synthetic**
(model-generated). Good news up front: **synthetic data in the active
pipeline is genuinely minimal — one touchpoint, not several** — and
it's already been deliberately reduced from a larger footprint in an
earlier PRD revision.

## What the repo already removed (verified, not just claimed)

`docs/data-forge/ARCHITECTURE.md` states directly: *"Every step that
previously depended on synthetically-generated or AI-judge-labeled
data has been removed and replaced with either a real, already-
published, human-labeled dataset or a training-free inference-time
technique."* Confirmed in code, not just in the doc: `pipeline.yaml`
has an explicit removed-stage marker (`s07_5_edit_pairs`, "no-RLHF-loop
/ no-synthetic-data revision") and `tier1.py` explicitly states *"this
pipeline [no longer] generates synthetic conversational content"* for
what used to be self-distilled Planner SFT data. Since the Planner and
Critic both ship frozen (Phases 3 and 4 of this review), neither has
any training data — synthetic or otherwise — at all.

## The one real synthetic-data touchpoint: `s05_recaption`

`data-forge/data_forge/stages/s05_recaption.py` runs a VLM
(`Tier1Engine.generate_caption`) over every safety-cleared `ui_first`
image to produce a denser caption than the source dataset's own. This
**is** synthetic text data — model-generated, not human-written — and
it feeds directly into Sketch tier training's text conditioning
(`prompt_dim=768` CLIP embeddings computed from these captions, per
Phase 1's findings).

**One real mitigating factor, verified in code, not assumed:** the
call passes `source_caption_hint=rec.source_caption` — the VLM is
*densifying/elaborating* the dataset's own original human-written
caption, not free-generating from the image alone. This meaningfully
lowers (doesn't eliminate) the risk relative to pure from-scratch VLM
captioning, since the real caption's actual content anchors the
output.

**What this means for generalization — the actual risk to name in the
paper:** the Sketch tier's text-conditioning distribution at *training*
time is shaped by one specific VLM's captioning style (its typical
vocabulary, phrase structure, level of detail) layered on top of real
source captions. At *inference* time, real users will type their own
prompts — plain, inconsistent, sometimes terse, not VLM-style dense
descriptions. This is a genuine **train/inference distribution
mismatch risk**, not a data-quality problem — the images and the
underlying caption content are real, but the *phrasing distribution*
the model learns to condition on isn't identical to what it'll see at
serve time. This is worth stating explicitly in the paper's
limitations section, and worth a concrete pre-launch check: run a
batch of realistic, plainly-worded user-style prompts (not VLM-style
dense ones) through the trained Sketch tier and confirm output quality
doesn't degrade — not yet done in this review (no trained checkpoint
exists to test against).

## Everything else, checked and confirmed real

- **RICO (core+semantic), CLAY, Enrico, WebUI, Screen2Words, UICrit** — human-created UI screens with human-written/human-authored annotations. No synthetic-generation step found anywhere in their ingestion path.
- **PD12M, CC12M** (general-domain images for the Polish tier's broader visual grounding) — real photographs; captions are web-sourced alt-text (CC12M) or public-domain metadata (PD12M), not model-generated.
- **Pick-a-Pic v2, HPDv2 (DPO preference pairs)** — real human pairwise preference judgments. Checked `s01_6_preference_pairs.py` and the surrounding `dpo/` package directly for any auto-labeling / reward-model-generated preference signal: **found none.** Every preference label in the active pipeline traces to a real human comparison.
- **DesignSense-10k, DesignPref** (UI-domain preference pairs, Stage 2) — real human (professional designer) comparisons, per their own papers — moot for now since, per Phase 2's findings, both are confirmed unreleased and Stage 2 cannot actually run yet.

## Generalization coverage — summary

| Data type | Real or synthetic | Generalization risk |
|---|---|---|
| Sketch tier images (RICO/CLAY/Enrico/WebUI) | Real | Low — genuine UI screen diversity across 4 independent sources |
| Sketch tier captions | **Synthetic (VLM-densified, real-caption-anchored)** | **Medium — train/inference prompt-style mismatch; flagged above as the one real risk to test and disclose** |
| Polish LoRA images/captions (PD12M/CC12M + UI sources) | Real | Low |
| DPO preference labels (Pick-a-Pic/HPDv2) | Real | Low — but general-domain, not UI-domain (Stage 2's UI-domain data is unreleased, so current DPO alignment generalizes from general image preferences, not design-specific ones — a scope limitation worth naming, distinct from a synthetic-data risk) |
| Planner grounding (UICrit RAG corpus) | Real | Low |
| Critic (frozen, zero-shot) | N/A — no training data | N/A |

**Bottom line for the paper:** you can accurately write that this
system trains on real, human-created data almost throughout, with
exactly one deliberate, disclosed exception (VLM-assisted caption
densification, anchored to real source captions) — and that exception,
plus the separate general-vs-UI-domain DPO scope limitation, are the
two honest generalization caveats worth a sentence each in a
limitations section, rather than something to gloss over.

---
See `docs/review/05_data_pipeline.md` for the broader data-pipeline
correctness review this audit builds on, and `docs/review/01_sketch_tier.md`
/ `02_polish_tier.md` for how each dataset is actually consumed by
training.
