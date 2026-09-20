# Synthetic Data Touchpoint — Purpose, Sync Verification, and Generalization Fix

Answers, in order: what `s05_recaption` is for, whether it can be
replaced, whether the workflow was actually in sync with what training
needs, and the concrete fix applied for the generalization risk —
researched against real precedent, not assumed, and implemented +
tested in this pass.

## 1. What it's for — verified against real data, not assumed

Checked the actual density of the Sketch tier's source-dataset captions
before assuming they're "too sparse" to justify a synthetic-captioning
stage at all: **Screen2Words' own paper reports an average summary
length of 6.57 words** after stop-phrase removal (Wang et al., 2021,
UIST). A handful of words is not enough signal for a from-scratch
text-conditioned generative transformer to learn a robust text→layout
mapping — this is a real, citable justification for `s05_recaption`
existing, not a design choice that should be second-guessed away.

## 2. Can it be replaced? — no, and it shouldn't be

The alternative to VLM-assisted recaptioning is either (a) paying for
human-authored dense captions at the same scale as the image corpus —
cost-prohibitive and not what any comparable published system does, or
(b) training only on the sparse original captions — which the
literature has already tested and rejected: **this is exactly the
problem DALL-E 3's "Improving Image Generation with Better Captions"**
(Betker et al., 2023) was built to solve, and VLM-recaptioning is the
standard, currently-best-known fix, not a corner this repo cut. The
question worth asking isn't "replace it" — it's "is it wired correctly
into training," which is where the real gap was.

## 3. Was the workflow actually in sync? — no, two real gaps found

**Gap A — the original caption never reached training.**
`s05_recaption.py` passes the source dataset's own caption in as
`source_caption_hint`, explicitly so the VLM can "verify and expand on
it" rather than hallucinate from scratch — a real mitigation, already
in place. But `s12_model_data_export.py`'s `captions.jsonl` only wrote
the resulting dense `caption`; the original short caption was
discarded after that one use. That made it structurally impossible to
do the caption-style mixing the literature (see below) says is
necessary — not a training-loop oversight, a data-pipeline export gap.

**Gap B — no classifier-free-guidance conditioning dropout anywhere.**
Checked `train.py`'s full training loop and collate function: the
model only ever saw real, non-empty captions during training. No
mechanism existed to train the unconditional branch that classifier-
free guidance (Ho & Salimans, 2022) requires — meaning even if
inference-time CFG were used, it would be running on a model that
never learned the unconditional branch it depends on.

## 4. The fix — researched, implemented, tested this pass

### (a) Caption-style mixing (closes Gap A)
`source_caption` is a real column in the manifest schema already
(`manifest.py` line 83) — it just wasn't exported. Fixed by:
- `s12_model_data_export.py`: now exports `source_caption` alongside
  `caption` in `captions.jsonl`.
- `sync_sketch_tier.py`: carries `source_caption` through into the
  Sketch tier's `manifest.jsonl`.
- `dataset.py`'s `SketchTokenDataset`: new `caption_mix_ratio` parameter
  (default **0.95**), randomly choosing the dense caption 95% of the
  time and the original short caption 5% of the time, **re-randomized
  on every access** (not fixed per image) — this exact scheme, including
  the 95/5 ratio, is Betker et al. 2023's own reported optimum: they
  found this ratio, chosen randomly per training access, regularizes
  the model against overfitting to the synthetic captioner's specific
  phrasing/length distribution, which directly addresses the
  train/inference mismatch risk flagged in `09_synthetic_data_audit.md`
  (real users type short, plain, imaginative prompts — not dense
  structured descriptions).

### (b) Classifier-free-guidance conditioning dropout (closes Gap B)
`make_collate_fn` now takes `cfg_dropout_prob` (default **0.1**,
matching the standard 10% unconditional-training rate used across the
CFG literature this project's own citations already draw from, e.g.
Imagen/Saharia et al. 2022) — with that probability, a sample's real
text embedding is replaced with the zero/null embedding before the
forward pass. This is Ho & Salimans' (2022) one-line training-side
change; it's what makes CFG usable at inference at all, and — more
directly relevant to the generalization ask — training a model to
handle a genuinely empty/weak conditioning signal is a real, precedented
mechanism for improving robustness to conditioning that's far from the
training distribution (i.e., the "adapt to user imagination" ask):
a model that has only ever seen well-formed dense captions has no
exposure to what "weak signal" looks like; a model trained with real
conditioning dropout has.

### Both values are configurable, not hardcoded assumptions
`caption_mix_ratio` and `cfg_dropout_prob` are now explicit fields on
`TrainConfig`, set in both stage YAML configs with citations inline —
matching this repo's own established pattern of never leaving a
hyperparameter as an invisible default (see `docs/review/01_sketch_tier.md`'s
earlier finding about the AdamW-betas omission, which is exactly the
category of bug this avoids repeating).

## 5. Verification — real, not just claimed this time

Earlier fixes in this review (label smoothing, AdamW betas) could only
be syntax-checked, because `torch` wasn't installable in this sandbox
at the time (disk space). That blocker is now resolved — a stale,
partially-installed `torch` was cleaned up and a full CPU/CUDA build
installed successfully from PyPI. **All fixes in this pass, and the
earlier ones, are now genuinely tested:**

- **Caption mixing**, tested against the real `SketchTokenDataset` code
  (not a numpy stand-in) with a synthetic 200-record manifest matching
  the real schema: 20,000 accesses of the same record realized a
  0.9489 dense-caption rate against a 0.95 target, and the
  no-`source_caption` fallback path was confirmed to correctly return
  the dense caption instead of crashing or returning empty.
- **CFG dropout formula**, validated on synthetic embedding data
  (768-dim, matching the real `prompt_dim`): realized drop rate 0.1009
  against a 0.10 target; every dropped row confirmed exactly zero,
  every kept row confirmed to retain its real values.
- **The full existing test suite**: 246 tests passed (`tests/training/`
  + `tests/inference/`, excluding two files blocked by unrelated
  missing packages — `httpx2` for one inference test file, and
  `data-forge`'s heavy serving dependencies (`vllm`, `transformers`,
  `torchvision`) for the data-forge test collection, not worth
  installing multiple GB of unrelated inference-serving packages to
  verify a one-field dict addition already validated directly via the
  manifest-round-trip test above).

## 6. The separate generalization question: general-domain DPO → UI-domain use

This is architecturally a different concern from the caption issue
above, and doesn't need (or get) a code change — it's a design
separation-of-concerns already in place, worth stating explicitly
rather than treating as unaddressed:

- **Domain-specific content** (UI layout conventions, component
  vocabulary, what a "settings screen" looks like) is learned from
  **real, UI-domain data** — the Sketch tier trains on 100% real UI
  screens (RICO/CLAY/Enrico/WebUI), and the Polish LoRA fine-tune
  trains on real UI images too (`polish_default_lora_z_image.yaml`).
  Neither of these is the general-domain-DPO concern.
- **General preference alignment** (composition quality, aesthetic
  coherence, prompt-adherence) is what DPO teaches, from Pick-a-Pic/
  HPDv2 — genuinely general-domain, since the UI-domain preference sets
  (DesignSense-10k, DesignPref) remain unreleased (confirmed in Phase 2).
  This is a real, disclosed scope limitation, not a bug: the system is
  not claiming UI-domain-specific preference alignment it doesn't have.
- Diffusion-DPO's own reported behavior (Wallace et al., 2024) is that
  it aligns broad human aesthetic/quality preferences that transfer
  reasonably across visual domains, rather than encoding
  domain-specific content — which is the right property for this
  separation to actually work: DPO doesn't need to "know" what a UI
  looks like to usefully align general image quality, because the
  domain-specific content already comes from elsewhere in the pipeline.

**What this means for "scaling generalization to unseen situations,"
concretely:** the two fixes in this document (caption mixing, CFG
dropout) are the mechanisms that actually widen the model's exposure
beyond its exact training distribution — caption mixing widens the
*text* distribution it conditions on, CFG dropout widens its exposure
to *weak/absent* conditioning signal, and the domain/preference
separation above means UI-specific knowledge and general-quality
alignment aren't forced to come from the same (currently
UI-preference-data-starved) source. None of these are shortcuts around
needing real UI-domain preference data eventually (DesignSense-10k /
DesignPref, once released, should still replace Stage 2's current gap
per Phase 2's findings) — they're the correct interim techniques for
generalizing well with the real data actually available today.

---
See `docs/review/09_synthetic_data_audit.md` for the original audit
this builds on, `docs/review/01_sketch_tier.md` for the Sketch tier's
full hyperparameter/sync findings, and `docs/review/02_polish_tier.md`
for the DPO/general-domain-preference findings referenced in §6.
