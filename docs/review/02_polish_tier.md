# Phase 2 — Polish Tier (Z-Image LoRA fine-tune + Z-Image-Turbo Diffusion-DPO)

## What it is
Two separate trained artifacts under "Polish":
1. **`polish_default` LoRA** — dreambooth-style LoRA fine-tune of the *undistilled* `Tongyi-MAI/Z-Image` base via diffusers' official `train_dreambooth_lora_z_image.py`, served at inference by loading that LoRA onto the distilled `Z-Image-Turbo` checkpoint.
2. **Z-Image-Turbo Diffusion-DPO** — LoRA + flow-matching-adapted Diffusion-DPO alignment, directly on `Tongyi-MAI/Z-Image-Turbo`, in two stages (general preference → UI-domain preference, gated on UI-domain data actually being released).

## Data-forge ↔ Training sync

### 1. `polish_default` LoRA — clean, no mismatch
`sync_polish_default.py` reads `model_data/polish_zimage_turbo/{images/,captions.jsonl}` and does a **straight format bridge** (record-keyed → filename-keyed captions) into `dataset_prep.prepare()`'s expected layout — no re-encoding, no tokenizer, because the official diffusers script computes its own latents internally from raw images. This matches `SYNC_DESIGN.md`'s stated reason #2 and is **in sync**, same class of "real, justified bridge" as the Sketch tier's tokenizer workaround, minus the encoder mismatch — there's no encoder here to mismatch.

One real bug that *was* caught and fixed in-repo (worth citing as a template for reviewing the DPO path too): captions used to be joined on `record_id == filename.stem`, which is always false for a real export (record_id is a uuid4, filenames are scrubbed originals). Fixed by having `s12_model_data_export.py` emit `image_filename` explicitly and joining on that. Confirmed present and correct in the current `sync_polish_default.py` and `sync_sketch_tier.py` both.

### 2. Diffusion-DPO — real trainer exists, but one docstring is now stale
- `training/dpo/` (pair building/export) is intentionally scoped as **not** the training loop — its own `__init__.py` says so, and that's accurate as a description of that subpackage's boundary, not a project-wide gap.
- The actual trainer is `training/polish/train_dpo.py` + `dpo_loss.py`, which **do** exist, are structurally complete (optimizer, grad accumulation, reference-model freezing, checkpointing, LoRA via `add_adapter`), and explicitly self-describe as closing "the previously-flagged gap."
- **Stale doc found**: `training/data_forge_bridge/sync_dpo_pairs.py`'s docstring still says *"this project has no DPO trainer yet that consumes Z-Image-Turbo latents directly (see `dpo/__init__.py`)"* — that's no longer true; `train_dpo.py` is exactly that trainer. The underlying design decision the docstring is defending (import raw images via BlobStore rather than data-forge's precomputed latents) is still correct — `train_dpo.py` resolves `chosen_ref`/`rejected_ref` through the shared `BlobStore`, i.e. it also consumes images, not `s08_5_dpo_encoding.py`'s `.safetensors` — so nothing is functionally broken, but the docstring's stated reason ("no trainer exists") is now false and should say "the real trainer also consumes images via BlobStore, not precomputed latents" instead.
- **Confirmed orphaned artifact, same shape as Phase 1's `maskgit_vq` finding**: `s08_5_dpo_encoding.py`'s `.safetensors` latent output is referenced *only* in comments explaining why nothing reads it — grep across `training/` and `inference/` shows zero real consumers. Not broken (unlike `maskgit_vq`, this encoder presumably loads fine — not verified in this pass), just genuinely unused. Recommend either wiring a latent-consuming fast path into `train_dpo.py` (real speed win if DPO training becomes GPU-time-constrained) or removing the stage to stop maintaining a producer nothing consumes.

### 3. Source-key consistency — in sync
`preference_store.VALID_SOURCES` (`pickapic_v2`, `hpdv2`, `designsense_10k`, `designpref`, plus three internally-generated sources) matches `sync_dpo_pairs.py`'s `_SOURCE_KEY_MAP` 1:1, which matches `dpo_z_image_stage1_general.yaml`'s `source: ["pickapic_v2", "hpdv2"]` and the Stage-2 override comment (`designsense_10k`, `designpref`) — and correctly reflects `DATA_SOURCES.md`'s finding that the latter two have **no public release yet**, so Stage 2 cannot actually run today. This is consistent end-to-end, not a gap — the repo is honest that Stage 2 is blocked on external data availability, not on its own code.

## Hyperparameters vs. precedent
- LoRA fine-tune: resolution 1024, batch 1 × grad-accum 4, lr 1e-4 constant schedule, 8-bit AdamW, bf16, 1000 steps — standard dreambooth-LoRA defaults, nothing unusual.
- DPO: `beta=2000.0`, `fm-anchor-weight=0.0`, lr 1e-5, LoRA rank 32, resolution 512 (lower than the LoRA fine-tune's 1024 — reasonable for DPO's typically smaller/cheaper preference-comparison batches), 1000 steps. **The repo itself flags these as untuned starting points** (`dpo_z_image_stage1_general.yaml`'s header comment, `train_dpo.py`'s docstring) rather than validated values — appropriately conservative self-assessment, not a finding against the repo, but worth stating plainly in the paper: no real hyperparameter sweep has been run (no GPU access in this environment).
- `beta=2000.0` is unusually large relative to the Wallace et al. Diffusion-DPO paper's own reported range (typically β in the few-thousand range *is* actually what that paper uses for image diffusion, unlike LLM-DPO's much smaller β — worth double-checking the exact figure directly against Wallace et al. Table/Eq. 46 before citing a specific number in the paper; flagging as **needs primary-source verification**, not confirmed wrong).

## Inference strategy vs. SOTA
- The core adaptation — substituting velocity-prediction squared-error terms for epsilon-prediction terms inside Wallace et al.'s sigmoid-of-difference-of-differences DPO loss, because Z-Image-Turbo is flow-matching/rectified-flow (flux-dev lineage) rather than DDPM-epsilon — is not an invented shortcut; it mirrors a published adaptation (MotionFlux, Bin et al., arXiv:2508.19527, §3.6), including that paper's anchor-regularization term for the known flow-model DPO failure mode (drift away from the pretraining distribution via reward hacking). This is a **citable, defensible design choice**, correctly attributed in-repo.
- Serving-side: `polish_default_backend.py` loads the LoRA onto the distilled Turbo checkpoint rather than the undistilled base it was trained on — standard, low-risk practice for LoRA (small-rank deltas transfer reasonably across a base→distilled pair sharing weights), but this is an assumption, not something verified against Z-Image-Turbo's actual distillation method in this pass. **Flagging as unverified, not wrong** — recommend a quick qualitative eval (render the same prompt with/without the LoRA on both base and Turbo) before finalizing this phase.

## Diffusion convergence hyperparameters — verified against current SOTA
(This is the phase where "diffusion" actually applies — the Sketch tier
is a non-diffusion masked-token transformer.)

`train_dpo.py` uses `diffusers.training_utils.compute_density_for_timestep_sampling(weighting_scheme="logit_normal", ...)`,
sampled independently for the chosen and rejected side of each pair —
**verified directly in the code, not assumed.** This is correctly
aligned with current best practice for flow-matching/rectified-flow
models: SD3, Flux, and Lumina-T2X all report logit-normal timestep
sampling converging measurably faster than uniform sampling, by biasing
training toward the harder mid-range timesteps where velocity-prediction
difficulty peaks (Esser et al., SD3; Lumina-T2X, arXiv:2405.05945). The
repo already does this correctly — a real, citable, SOTA-aligned choice.

One caveat from the literature worth stating in the paper rather than
treating logit-normal as unconditionally optimal: a 2026 study
(arXiv:2603.12517) shows logit-normal sampling accelerates *early*
convergence but under-samples trajectory boundaries (t→0, t→1),
imposing a *ceiling* on final quality relative to a curriculum that
shifts toward uniform sampling later in training. Not a bug in this
repo — just a real trade-off to name explicitly rather than imply
logit-normal alone is strictly best end-to-end.

Still-open numeric items from the hyperparameters section above stand
unchanged: `beta=2000.0` needs verification against Wallace et al.'s
actual reported range, and DPO's `resolution=512` vs the base LoRA's
`1024` is a reasonable but unconfirmed cost/quality trade-off.

## Findings summary

| Severity | Finding |
|---|---|
| Info | `polish_default` bridge is clean — no encoder mismatch, correct filename-join fix already in place |
| **Doc bug** | `sync_dpo_pairs.py`'s docstring claims "no DPO trainer yet" — false, `train_dpo.py` exists and closes exactly that gap. Underlying design (import images, not latents) is still correct; only the stated justification is stale. |
| **Real gap** | `s08_5_dpo_encoding.py`'s `.safetensors` output has zero real consumers anywhere in `training/` or `inference/` — same orphaned-stage pattern as Phase 1's `maskgit_vq`. Not confirmed broken, just unused. |
| Resolved | `beta=2000.0` — **verified correct**, not just plausible. Loss implementation (`dpo_loss.py`) matches Wallace et al.'s Eq. 46 normalization exactly (per-sample mean MSE, same margin structure), and beta=2000 is independently confirmed as the paper's actual reported value (corroborated via arXiv:2410.05255: "we set β=2000, which stays the same in Diffusion-DPO"). Correct only *because* the implementation matches that scale — worth stating that reasoning explicitly in the paper, since other DPO-diffusion papers report very different "optimal" β on different loss normalizations. |
| Resolved (corrected) | ~~LoRA rank is set for DPO but not for the base fine-tune, and both get composed at inference~~ — **the "composes both" part was wrong**, corrected during the scripts/ review (`inference/backends/factory.py` only ever loads ONE LoRA path for `POLISH_DEFAULT`, via `KRISNA_POLISH_DEFAULT_LORA_PATH`; `dpo_z_image_stage1_general.yaml`'s `lora-adapter-path: null` default means DPO trains independently from raw base weights unless an operator explicitly chains it, and the DPO checkpoint is meant to *supersede* the base LoRA at deploy time by pointing the same env var at its output — not run alongside it). The rank-pinning fix (`rank: 32` on the base config) is still good practice for the case an operator *does* chain DPO onto the base checkpoint via `lora-adapter-path`, but the capacity-mismatch risk is lower-severity than originally stated, since by default only one adapter is ever deployed at a time. |
| Action | Confirm LoRA-on-distilled-base serving assumption with a real qualitative check before finalizing |

## Citations (Phase 2)

1. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., & Naik, N. (2024). *Diffusion Model Alignment Using Direct Preference Optimization*. CVPR 2024.
2. Bin, Y., et al. (2025). *MotionFlux: [flow-matching DPO adaptation for motion generation]*. arXiv:2508.19527, §3.6.
3. Kirstain, Y., et al. — Pick-a-Pic v2 dataset (`yuvalkirstain/pickapic_v2`, HuggingFace, MIT license).
4. Hao, Y., et al. — HPDv2 dataset (`ymhao/HPDv2`, HuggingFace, Apache 2.0), ~798K pairwise comparisons / ~434K images.
5. DesignSense-10k — Adobe MDSR, arXiv:2602.23438 (UI/design-domain preference pairs, CC BY-NC-ND 4.0, **no public dataset release found** as of this review — cite as "referenced, not trainable on" if used in the paper).
6. DesignPref — Peng et al., 25 Nov 2025 (UI-domain pairwise comparisons, ~12,000 pairs, 20 professional designers, **confirmed unreleased** per TASTE's own paper, arXiv:2605.20731).

---
Next: Phase 3 — Planner (Qwen3.5-9B, frozen + RAG — no training to review, but the RAG corpus/retrieval design itself needs checking against what UICrit data-forge actually exports).
