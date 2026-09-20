# Krisna Training

> **Split from the original krisna-orchestrator README** as part of the
> training/inference package separation — see
> `../docs/architecture/DIRECTORY_LAYOUT.md`. This covers everything
> that actually trains a checkpoint, plus the data-forge sync bridge and
> DPO preference-pair data prep. For the serving/orchestration side
> (`krisna_inference`), see `../inference/README.md`.

## Architecture overview

`krisna_training` trains exactly **two** of the five components in the
PRD's model stack — everything else in the stack ships frozen (see the
root README's summary, and `../docs/architecture/RESEARCH_AND_CITATIONS.md` §1 for the
full reasoning + citations behind that split):

| Subpackage | Trains | Method | Status |
|---|---|---|---|
| `sketch/` | Sketch tier (MaskGIT-lineage) | From scratch, two-stage progressive resolution | **Active** |
| `polish/` | Z-Image-Turbo | LoRA fine-tune (+ Diffusion-DPO via `dpo_dataset.py`) | **Active** |
| `preference/` | — (not a model) | Builds/exports DPO preference-pair data from data-forge's real, human-labeled pairs | **Active** — canonical replacement for legacy `dpo/` |
| `data_forge_bridge/` | — (not a model) | Re-tokenizes/re-encodes data-forge's raw images through this package's own working tokenizers | **Active** — the only path real training data reaches this package |
| `dpo/` | — (not a model) | Backward-compatibility forwarder to `preference/` | **Active (Legacy alias)** |
| `planner/` | Qwen3.5-9B Planner | BF16 LoRA chat-SFT | **Deprecated** — Planner ships frozen (RAG retrieval instead, see `inference/README.md`) |
| `critic/` | Gemma 4 31B Critic | QLoRA via Unsloth | **Deprecated** — Critic ships frozen (on-demand product feature) |

Deprecated subpackages emit a `DeprecationWarning` on import and are kept
as real, working reference code — not deleted — in case a future PRD
revision un-freezes either tier. See each one's own `__init__.py`
docstring for exactly what to revert if that happens.

## Complete walkthrough — from data-forge output to a trained checkpoint

This is the real, end-to-end sequence, tier by tier. Every command below
is copy-pasteable from the monorepo root. On Windows, native PowerShell scripts
(`.ps1`) are available alongside the bash scripts (`.sh`).

**1. Run data-forge first** (see `../data-forge/README.md`) — this
produces `<DATA_ROOT>/model_data/` and `<DATA_ROOT>/preference_pairs/`,
the two directories every sync script below reads from.

**2. Sketch tier** (from scratch, real UI screenshots):
```bash
# Bash (Linux / Git Bash)
./scripts/training/setup_env_training.sh
./scripts/training/download_vqgan.sh
./scripts/training/sync_from_data_forge_sketch.sh <data_forge_model_data_dir> ./data/sketch_train_256
./scripts/training/train_sketch_stage1.sh   # 256px, from scratch (configs/sketch_stage1_256.yaml)
./scripts/training/train_sketch_stage2.sh   # 512px, progressive init from stage 1 (configs/sketch_stage2_512.yaml)

# PowerShell (Windows)
.\scripts\training\setup_env_training.ps1
.\scripts\training\download_vqgan.ps1
python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>
.\scripts\training\train_all.ps1 -Tier sketch-stage1
.\scripts\training\train_all.ps1 -Tier sketch-stage2
```

**3. Polish tier (Z-Image-Turbo)** — base fine-tune, then optionally DPO
alignment on top:
```bash
# Bash (Linux / Git Bash)
./scripts/training/setup_env_diffusers_training.sh
./scripts/training/sync_from_data_forge_polish.sh <data_forge_model_data_dir> ./data/polish_default_train
./scripts/training/train_polish_default_lora.sh  # configs/polish_stage1_default_lora.yaml

# PowerShell (Windows)
.\scripts\training\setup_env_diffusers_training.ps1
python scripts/data-forge/sync_to_training.py --data-root <DATA_ROOT>
.\scripts\training\train_all.ps1 -Tier polish-default

# Optional: DPO preference data, prepped from data-forge's real pairs
./scripts/training/sync_from_data_forge_dpo.sh <data_forge_data_root>
./scripts/training/train_polish_dpo.sh  # configs/polish_stage2_dpo_general.yaml
# Or in PowerShell:
.\scripts\training\train_all.ps1 -Tier polish-dpo
```

**4. Point the inference layer at what you trained** — see
`../inference/README.md`'s "Enabling real backends" section for the
`KRISNA_*` env vars that wire `models/` checkpoints into the actual
serving orchestrator, and `../inference-runtime/README.md` for a CLI to
smoke-test the result without standing up the full service.

## Per-tier detail

## Sketch tier training (closes the "no checkpoint" gap)

`training/src/krisna_training/sketch/` is real, working training code
for the sketch tier — the piece the README used to just call "genuinely
still missing." It does not change the inference-time sampling contract
in `inference/src/krisna_inference/backends/maskgit_model.py`; it's the concrete architecture (in
`model.py`) and training loop that produces checkpoints satisfying that
contract.

```bash
./scripts/training/setup_env_training.sh
./scripts/training/download_vqgan.sh                                    # boris/vqgan_f16_16384
./scripts/training/prepare_sketch_dataset.sh <image_dir> ./data/sketch_train_256 256
./scripts/training/train_sketch_stage1.sh                                # 256px, from scratch
./scripts/training/train_sketch_stage2.sh                                # 512px, progressive init from stage 1
```

Then point the inference layer at the result:
```bash
export KRISNA_SKETCH_CHECKPOINT=./checkpoints/sketch_stage2_512/checkpoint_final.pt
```

| Piece | What it is |
|---|---|
| `vq_tokenizer.py` | Wraps `boris/vqgan_f16_16384` — a real, publicly downloadable PyTorch VQGAN (github.com/CompVis/taming-transformers). NOT UI-domain-tuned; documented as the honest tradeoff (real > fabricated). Also the real implementation behind `inference/src/krisna_inference/backends/sketch_handoff.py`'s VQ-decode handoff hook, closing that previously-documented gap. |
| `model.py` | The bidirectional transformer satisfying `maskgit_model.py`'s `.forward(tokens, mask, prompt_embedding) -> (logits, critic_scores)` contract. Small by design (~44M params default) — matches §6.1's own "small-scale, ImageNet-lineage precedent" framing, not an attempt at a foundation model. |
| `masking.py` | Training-time random masking via the cosine schedule (distinct from the Halton-ordered inference-time reveal). |
| `losses.py` | Masked-token cross-entropy + Token-Critic BCE, the two-headed objective the architecture needs. |
| `pos_embed.py` | Bicubic positional-embedding interpolation for progressive 256→512px training (§6.1). |
| `dataset.py` / `prepare_dataset.py` | JSONL-manifest dataset + a tokenization script for turning a directory of UI images into training data. |
| `checkpoint_io.py` | Saves `state_dict()` + config (not a pickled model object — see its docstring for why that matters). `inference/src/krisna_inference/backends/maskgit_model.py`'s loader was updated to match. |
| `train.py` | The actual training loop: AdamW, cosine LR + warmup, bf16 autocast, gradient clipping, checkpointing, resume, progressive-init support. |

**A real bug this training code caught in the earlier-built inference
layer, while writing its own tests:** the Halton-ordered sampler's
per-round reveal-count formula divided by `num_rounds` twice (schedule
already encoded it, then divided again), which on small grids or low round
counts could leave `mask_token_id` values un-revealed after the final
round — `sample()` was fixed to target a cumulative reveal count and force
full reveal on the last round. `../tests/training/test_maskgit_sample_e2e.py` locks
this in across grid sizes and round counts, including the edge cases
(`num_rounds=1`, non-power-of-2 grids) that originally exposed it.

**What this does NOT include:** an actual training dataset (point
`prepare_sketch_dataset.sh` at RICO/CLAY/Enrico or whatever UI corpus you
have — this doesn't bundle one), a UI-domain-tuned VQ tokenizer (uses the
general-purpose ImageNet/CC3M one, documented as a real, working default
rather than an unverified fabrication), and a genuine planner→sketch
prompt-embedding contract (training defaults to CLIP text embeddings of
captions, a reasonable placeholder — not the "continuous conditioning
embeddings" from the trained Planner tier §5's diagram ultimately implies,
since that cross-tier contract doesn't exist yet either).



- Training the sketch tier itself (see "Inference layer" above for what
  IS built vs. missing there).
- The two pipeline-specific API details flagged in `train_dpo.py`
  (`encode_prompt()`'s return shape, the transformer's forward-call
  signature) aren't independently verified against Z-Image-Turbo's real
  pipeline class — see the inline comments at those exact call sites.
- Real training dynamics for `train_dpo.py` against actual weights —
  the loss math is verified (see below), a live GPU run is not.

## DPO alignment — Diffusion-DPO for a flow-matching model

`training/src/krisna_training/polish/dpo_loss.py` +
`train_dpo.py` — a real, tested consumer for `dpo/`'s preference-pair
exports, closing what was previously an open gap in this project. See
`../docs/architecture/RESEARCH_AND_CITATIONS.md` §4.5 for the full
derivation: this is **not** the original Diffusion-DPO loss applied
unmodified (that formulation assumes DDPM-style noise prediction;
Z-Image-Turbo predicts a flow-matching velocity instead) — it's a
published, cited adaptation (MotionFlux, arXiv:2508.19527) that
substitutes velocity-prediction error for noise-prediction error inside
the same DPO structure, plus an optional flow-matching anchor
regularization term to guard against reward-hacking-style drift.

**Verified, via real mathematical tests, not just shape checks** (`../tests/training/test_dpo_loss.py`):
the loss equals exactly `log(2)` when the policy hasn't diverged from the
reference (the standard closed-form DPO sanity check), rewards/penalizes
preference alignment correctly in both directions, a real gradient step
measurably improves the chosen-side prediction, and the flow-matching
sign convention is self-consistent (predicting the target velocity and
taking one Euler step from the noised latent exactly recovers the clean
latent).

**Not verified — needs a real GPU run**: actual training dynamics against
real Z-Image-Turbo weights, whether the default `beta=2000.0` (carried
over from Wallace et al.'s SD1.5/SDXL-tuned range) transfers reasonably
to this model, and two pipeline-specific API details flagged directly in
`train_dpo.py`'s inline comments (`encode_prompt()`'s return shape, the
transformer's forward-call signature) that weren't independently checked
against Z-Image-Turbo's live pipeline code.

```bash
./scripts/training/sync_from_data_forge_dpo.sh <data_forge_data_root>
# Run via shell script or CLI (canonical config: polish_stage2_dpo_general.yaml):
./scripts/training/train_polish_dpo.sh training/configs/polish_stage2_dpo_general.yaml
# Or via installed console script:
# krisna-train-dpo --preference-db krisna_preference_pairs.db --blob-root krisna_blobs
# Stage 2 (UI-domain — DesignSense-10k/DesignPref) has zero usable data
# right now (see RESEARCH_AND_CITATIONS.md §2.8-2.9) — the yaml's
# `source` list is the only thing to change once either dataset is
# actually public.
```

## Polish tier training (Z-Image LoRA; Qwen-Image-Edit — honest gap)

`training/src/krisna_training/polish/` takes a different approach from
Sketch and Planner: it does **not** implement a from-scratch diffusion
training loop. Getting flow-matching/noise-schedule/VAE-latent details
right for models this recent, without a GPU here to verify against, is a
real risk — so this wraps diffusers' own **official, maintained** LoRA
training scripts instead of reinventing them.

```bash
# Setup diffusers training dependencies
./scripts/training/setup_env_diffusers_training.sh     # clones diffusers, installs example deps
# Or on Windows PowerShell:
.\scripts\training\setup_env_diffusers_training.ps1

# Prepare dataset into HuggingFace imagefolder format:
./scripts/training/prepare_polish_dataset.sh <image_dir> ./data/polish_default_train [captions.json]
# Or in PowerShell:
.\scripts\training\prepare_polish_dataset.ps1 -ImageDir <image_dir> -OutputDir ./data/polish_default_train -CaptionsPath [captions.json]

# Train Z-Image DreamBooth LoRA (canonical config: polish_stage1_default_lora.yaml):
./scripts/training/train_polish_default_lora.sh
```

**Z-Image (polish_default): real, working.** Wraps
`train_dreambooth_lora_z_image.py`
(github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_z_image.py),
diffusers' own maintained script. `configs/polish_stage1_default_lora.yaml`
(legacy alias: `configs/polish_default_lora_z_image.yaml`)
holds the run parameters; the shell script translates them into the
`accelerate launch` call. Base model is `Tongyi-MAI/Z-Image` — the
**undistilled** foundation model, not `Z-Image-Turbo`. This corrects an
earlier reference in this project's PRD conversation to a
"Z-Image-De-Turbo" checkpoint that could not be confirmed to exist under
that name; `Tongyi-MAI/Z-Image` is the real, confirmed model serving the
same role (full gradient signal, since Turbo's distillation is reportedly
unreliable to train against directly).
`inference/src/krisna_inference/backends/polish_default_backend.py`'s `ZImageTurboBackend` loads the
resulting LoRA onto the Turbo checkpoint for fast serving
(`KRISNA_POLISH_DEFAULT_LORA_PATH`) — same architecture, so it loads
without error, but that specific undistilled-trained-onto-Turbo
combination's output quality is **unverified**.

**Qwen-Image-Edit-2511 (polish_quality): genuinely no official script
exists yet.** Confirmed directly from a diffusers maintainer's reply
(github.com/huggingface/diffusers/discussions/12469): *"we don't have a
training script for Qwen-image-edit-2509 at the moment. The only option
... is to maybe adapt the kontext training script."* This project has
**not** fabricated a working training script to paper over that gap. Two
real, cited options instead (see `src/krisna_training/polish/__init__.py` for the
full explanation):
1. **DiffSynth-Studio** (github.com/modelscope/DiffSynth-Studio) —
   ModelScope's separate, actively maintained framework with documented
   LoRA/full training support for the Qwen-Image family.
2. Adapt diffusers' `train_dreambooth_lora_flux_kontext.py` (the closest
   architectural analog — image+text→edited-image) to Qwen-Image-Edit's
   model loading calls. Real adaptation work, not done here.

`src/krisna_training/polish/dataset_prep.py` (the standard HF `imagefolder`
convention: a flat directory + `metadata.jsonl` with `file_name`/`text`)
works for either path, independent of which trainer eventually consumes
it. And once you have a LoRA from either path,
`inference/src/krisna_inference/backends/polish_quality_backend.py`'s `QwenImageEditBackend` already
knows how to load it (`KRISNA_POLISH_QUALITY_LORA_PATH`) — the gap is
producing the adapter, not serving one you already have.

## Critic training — DEPRECATED (Gemma 4 ships frozen)

`training/src/krisna_training/critic/` implements real QLoRA-via-
Unsloth training for Gemma 4 31B Dense, but **is not part of the active
pipeline under the final, no-RLHF-loop PRD revision** — the Critic ships
frozen as an on-demand product feature, never trained. Importing this
package emits a `DeprecationWarning`; see its `__init__.py` docstring for
what would need to change to revive it.

The code below is preserved as documentation of a real, working approach,
not as a current setup instruction:

<details>
<summary>Original (v10 PRD) instructions, for reference only</summary>

Must run inside `./venv-critic`, the same isolated environment
`critic_worker.py`/`critic_backend.py` use for the `transformers==5.5.0`
pin conflict (see "Critic tier isolation" above — that isolation reasoning
is unaffected by the freeze, since it's about version pins, not training).

```bash
./scripts/training/setup_env_critic.sh          # now also: pip install -e . --no-deps
./scripts/training/train_critic_qlora.sh
```

Dataset: JSONL, one record per example —
`{"image_path": "...", "constraints": {...}, "critique": {...}}`, where
`critique` matches the CritiqueResult shape (§5.2) minus the
inference-time provenance fields (`critique_source`, `raw_model_output_ref`
— not something the model should predict about itself).

Uses Unsloth's real, documented VLM LoRA API
(`FastModel.get_peft_model(..., finetune_vision_layers=, finetune_language_layers=,
finetune_attention_modules=, finetune_mlp_modules=, ...)` — confirmed via
docs.unsloth.ai/basics/vision-fine-tuning) with both vision and language
layers trained by default, since judging a finished render well needs
both perceiving the image and reasoning/writing about it.

The prompt used for training was checked against the exact inference-time
prompt automatically: `critic_worker.py` exposes `_build_prompt()` as a
standalone function specifically so `../tests/training/test_critic_dataset.py` can
import it directly and assert byte-for-byte equality against `dataset.py`'s
own copy.

</details>

## Syncing with data-forge

`training/src/krisna_training/data_forge_bridge/` bridges data-forge's
`model_data/` export (that project's `s12_model_data_export.py`) into the
exact input formats this project's training packages expect. Building
this surfaced two real format mismatches between the two projects — not
papered over, see `src/krisna_training/data_forge_bridge/__init__.py` for the full
explanation:

1. **The sketch tier's VQ tokenizer identity doesn't match.** data-forge's
   `vq_tokens/*.pt` files come from Open-MAGVIT2, whose loading code in
   data-forge's own `engine.py` raises `RuntimeError` on purpose — that
   wrapper isn't verified working yet. This project's sketch tier uses a
   different, real, working tokenizer (`boris/vqgan_f16_16384`). The sync
   does NOT consume data-forge's `.pt` tokens — it re-tokenizes
   data-forge's raw images (linked by a fix on the data-forge side, below)
   through this project's own tokenizer instead.
2. **Z-Image-Turbo's precomputed latents have no consumer** — the
   OFFICIAL `train_dreambooth_lora_z_image.py` script takes raw images,
   not precomputed latents. Same fix, same reasoning.
3. **DPO preference pairs are genuinely compatible** — no encoder
   mismatch, just a format bridge into `src/krisna_training/dpo/preference_store.py`.

**A small, real fix was also needed on the data-forge side**: its export
stage linked only *processed* artifacts (`vq_tokens/`, `latents/`) and
never the raw images either bridge above actually needs — so neither
folder had a working consumer on this side. Fixed there (adds `images/`
alongside each), plus a genuine pre-existing bug this caught along the
way: when a domain had zero matching records, the export never created
its own output directory before writing `captions.jsonl`, raising
`FileNotFoundError`. Both fixes are covered by new tests on the
data-forge side.

**A second, more serious bug was found and fixed on THIS side while
verifying the images/ fix above actually worked end-to-end**: both
`sync_sketch_tier.py` and `sync_polish_default.py` joined
data-forge's `captions.jsonl` (keyed by `record_id`, a random `uuid4`)
back to the linked image files by assuming `image_filename.stem ==
record_id` — a false assumption for every real data-forge export, since
`record_id` and the linked filename come from completely unrelated
sources (manifest UUID vs. original fetch-time filename). This produced
**empty captions for 100% of synced records** with no error anywhere.
Fixed on the data-forge side by having `captions.jsonl` carry the actual
linked filename directly (`image_filename` field); fixed on this side by
joining on that field instead of guessing, with a loud warning + graceful
degrade for pre-fix data-forge exports. The bug was masked by this
project's own tests, whose fixtures baked in the same false assumption
the code made (`f"{record_id}.png"`) — fixed fixtures now use
deliberately unrelated record_id/filename pairs, matching real data-forge
output. See `TestRecordIdVsFilenameBugFix` in both
`../tests/training/test_sync_sketch_tier.py` and `../tests/training/test_sync_polish_default.py`.

```bash
./scripts/training/sync_from_data_forge_sketch.sh <data_forge_model_data_dir> ./data/sketch_train_256
./scripts/training/sync_from_data_forge_polish.sh <data_forge_model_data_dir> ./data/polish_default_train
./scripts/training/sync_from_data_forge_dpo.sh <data_forge_data_root>   # note: DATA_ROOT, not model_data_root — reads the raw pre-latent-encoding stage
```

`preference_store.VALID_SOURCES` was extended with data-forge's exact
source keys (`pickapic_v2`, `hpdv2`, `designsense_10k`, `designpref`) so
imported pairs stay traceable to their real origin rather than being
collapsed into a generic label. Note `VALID_SOURCES` also still contains
`"gemma_critique"` — a real, not-yet-resolved leftover from the v10
architecture, where the Critic generated its own training preference
labels. Under the final PRD, no AI-judge-labeled data is generated
anywhere in this pipeline; that source key existing doesn't mean anything
currently writes it, but it's worth removing in a future pass rather than
leaving the data structure able to represent a pattern the project
explicitly rules out.

## CLI Entry Points

When `krisna-training` is installed (`pip install -e training/`), the following console commands are available directly:

- `krisna-train-sketch --config <path-to-yaml>`: Run MaskGIT sketch tier training loop.
- `krisna-train-dpo --preference-db <path> --blob-root <path> [options]`: Run Diffusion-DPO alignment trainer.
- `krisna-sync-rag --source <data-forge-model-data> --dest <target-data-dir>`: Synchronize UICrit RAG corpus for the frozen planner.

## Canonical Configuration Layout

Canonical configs reside in `training/configs/` with clear stage-tiered naming:
- `sketch_stage1_256.yaml`: 256px MaskGIT training from scratch.
- `sketch_stage2_512.yaml`: 512px progressive continuation with interpolated positional embeddings.
- `polish_stage1_default_lora.yaml`: Z-Image DreamBooth LoRA training.
- `polish_stage2_dpo_general.yaml`: Flow-matching Diffusion-DPO alignment.
- `deprecated_planner_lora.yaml` / `deprecated_critic_qlora.yaml`: Reference configs for deprecated tiers.

Legacy config filenames (`sketch_train_stage1_256.yaml`, `sketch_train_stage2_512.yaml`, `polish_default_lora_z_image.yaml`, `dpo_z_image_stage1_general.yaml`) are fully retained for backward compatibility.

