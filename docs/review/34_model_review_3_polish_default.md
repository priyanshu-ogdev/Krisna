# 34 — Model 3/5 deep review: Polish-Default (Z-Image-Turbo)

Systematic review of the Polish-Default tier across all three seams:
data-forge → training, training → inference, and the real Z-Image-Turbo
model's documented behavior (verified via real network access, not
assumed). Found one critical bug (a feature completely inert in the
real pipeline despite correct unit tests), one real training-quality bug
(mismatched LoRA alpha/rank), and upgraded a previously-vague citation
with a real, found source plus a genuinely new open question.

## Critical bug: caption-mixing upgrade was completely inert in the real pipeline

Following the same Betker et al. 2023 reasoning already applied to the
Sketch tier (`caption_mix_ratio`), added the same mixing mechanism to
`training/polish/prepare_dataset.py` — `load_captions()` now returns
both dense and source captions, and `prepare()` mixes them at the same
0.95 ratio. Tested directly against `prepare_dataset.py`'s own JSONL
calling convention: all green.

**Then traced the actual production caller** (the discipline this
project's reviews have repeatedly needed): `sync_polish_default.py`
builds its own `captions_by_filename` dict by hand, BEFORE ever calling
`prepare()` — and it only stored the caption as a plain string,
discarding `source_caption` entirely. Since `load_captions()`'s
plain-dict branch has no slot for a paired source caption, **the entire
upgrade was completely inert for the real data-forge pipeline** — the
only production caller of this code — despite every direct unit test
of `prepare()`/`load_captions()` passing cleanly. This is the same
general risk category found in `docs/review/32`'s Sketch review
(parameter accepted but never wired at a seam between two components),
just manifesting at the data layer instead of the inference layer.

Fixed at the root: `sync_polish_default.py` now stores
`{"caption": ..., "source_caption": ...}` per filename, and
`load_captions()`'s dict-input branch was extended to accept this
richer nested shape (auto-detected per entry, so the original plain
`{filename: "caption string"}` calling convention still works
unchanged). Verified with a new end-to-end test
(`TestSourceCaptionSurvivesEndToEnd`) that calls the REAL `sync()`
entrypoint — not `prepare()`/`load_captions()` in isolation — at both
`caption_mix_ratio=0.0` and `1.0`, confirming the source caption
actually reaches the final `metadata.jsonl` file through the real call
chain. This is exactly the test shape that would have caught the bug
originally: it exercises the seam, not just each side of it.

## Real bug: LoRA alpha silently mismatched against rank

Both `polish_stage1_default_lora.yaml` and its verbose twin
`polish_default_lora_z_image.yaml` set `rank: 32` explicitly but never
set `lora_alpha` at all. Cloned the real
`huggingface/diffusers` repo and read
`examples/dreambooth/train_dreambooth_lora_z_image.py`'s actual argparse
directly: `--rank` and `--lora_alpha` both default to **4** — the
standard alpha==rank LoRA convention. Overriding rank to 32 without also
setting alpha left these configs silently training at **4/32 = 1/8th**
the conventional LoRA scaling: a real, no-error, quietly-weaker-than-
intended adaptation. This project had already found and fixed the exact
same class of gap once before, for the DPO stage's fresh-adapter
`LoraConfig` (`train_dpo.py`: `LoraConfig(r=args.lora_rank,
lora_alpha=args.lora_rank)`) — the base stage had the same gap in a
different form. Fixed both config files (`lora_alpha: 32`, matching
rank) and added `tests/training/test_polish_lora_config_consistency.py`
to lock this in, plus confirm the two config variants and the DPO stage
all agree with each other on rank — the exact kind of config-drift this
project's own review has caught before (e.g. the `caption_mix_ratio`
stale-default incident).

While in this file, also closed out a previously-unverifiable caveat:
an earlier review pass had written "confirm this flag name (`rank`)
matches... no network access to fetch third_party/diffusers here." This
pass had real network access — cloned the actual repo and confirmed
`--rank` is correct.

## Upgraded a vague citation with a real, found source — and a genuinely new open question

`polish_default_backend.py`'s docstring said training the LoRA against
undistilled `Tongyi-MAI/Z-Image` rather than `Z-Image-Turbo` directly
was "per the community-reported finding that Turbo's distillation
gradients are unreliable" — vague, uncited. Found the real source: a
Tongyi-MAI training-strategies writeup documenting that LoRA
fine-tuning directly on Turbo causes it to effectively "de-distill" —
generation quality measurably *improves* when reverting to the
non-accelerated 30-step/cfg=2 regime, meaning the 8-step acceleration
this project's whole Polish-Default tier exists for gets silently
destroyed by naive fine-tuning on Turbo. This project's actual approach
(train on the undistilled base) correctly avoids that specific failure
mode — now cited properly instead of gestured at.

The same source also names a different, more specialized technique —
"Differential LoRA" via a preset adapter — specifically designed to
customize Turbo while *preserving* its acceleration, which this
project's current approach doesn't directly target (it sidesteps the
de-distillation problem by training elsewhere, but doesn't verify the
resulting adapter preserves Turbo's distilled sampling quality as well
as a purpose-built Differential-LoRA approach might). Documented as a
genuinely new, flagged open question — extending PRD §11's own
open-questions list in spirit — not quietly resolved or silently
adopted without evaluation.

## Verified correct, no changes needed

- `polish_default_backend.py`'s `num_inference_steps=9`,
  `guidance_scale=0.0` — cross-checked against the real model card,
  official diffusers docs, and three independent community deployment
  recipes. All agree exactly.
- `ZImageImg2ImgPipeline`'s `strength=0.6` default — re-confirmed
  against the pinned diffusers commit's real source in an earlier pass;
  unaffected by this review.
- `resolution: 1024` in both base LoRA configs matches
  `polish_default_backend.py`'s `init_image.resize((1024, 1024))`.
- `lora-target-modules: "to_q,to_k,to_v,to_out.0"` in the DPO config —
  confirmed this is the official script's own real default set (verified
  directly from source), just explicit rather than implicit.

## Test coverage

11 new/updated tests in `test_polish_prepare_dataset.py`
(`TestSourceCaptionMixing`), 3 new end-to-end tests in
`test_sync_polish_default.py` (`TestSourceCaptionSurvivesEndToEnd`), 4
new config-consistency tests in
`test_polish_lora_config_consistency.py`. Full inference suite (211)
and training suite (193) both pass.
