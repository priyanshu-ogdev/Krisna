# Research Findings & Citations

Every non-obvious architectural decision in this repo traces back to a
specific, checked fact — not an assumption. This document is that trail:
what was claimed, what was actually verified, against what source, and
what changed as a result. Organized by decision area. Where a claim
couldn't be verified, that's stated too — several real gaps in this
project are "confirmed absent," not "assumed fine."

---

## 1. The no-RLHF-loop PRD revision — why five models became two

**Original (v10) design**: all five model-stack components (Planner,
Sketch tier, Polish Default, Polish Quality, Critic) were trained, with
the Critic (Gemma 4) generating its own preference labels for DPO —
AI-judge self-distillation.

**What changed and why**: the project's own review of this design
surfaced two structural problems with the Critic-generates-its-own-labels
approach: (1) it's a self-referential RLHF-style loop with no independent
ground truth, and (2) building a full QLoRA training pipeline for a
model whose only training signal is its own prior output is a lot of
machinery for a signal quality that can't be independently checked. The
revision: freeze the Planner, Polish Quality, and Critic entirely
(zero-shot ICL / RAG retrieval / on-demand product feature respectively),
and source DPO alignment for the one renderer that stays trained
(Z-Image-Turbo) from **real, human-labeled public preference datasets**
instead of self-generated critic output.

**Net effect**: of five original components, two are trained (Sketch
tier from scratch, Z-Image-Turbo via LoRA + Diffusion-DPO), three ship
frozen. See `docs/architecture/DIRECTORY_LAYOUT.md` for how this maps to
the actual package split.

---

## 2. Dataset verification — the DPO/RAG data sources

Every dataset below was checked against its live HuggingFace dataset
card or GitHub repo directly — not assumed from a paper title or a
plausible-sounding repo name. Several checks caught real, load-bearing
mismatches.

### 2.1 PD12M (`Spawning/PD12M`)

**Claim checked**: column names for the URL-list fetch path.
**Finding**: dataset card's documented schema is `id, url, s3_key,
caption, hash, width, height, mime_type, license, source` — `url` and
`caption` are exact matches, not a guess. **License**: CDLA-Permissive-2.0,
confirmed on the card.

### 2.2 CLAY (`google-research-datasets/clay`, GitHub)

**Claim checked**: license status (previously logged as "unclear,
archived repo, likely to be excluded").
**Finding**: the paper's own copyright footer states directly *"This work
is licensed under a Creative Commons Attribution International 4.0
License,"* and the umbrella `google-research/google-research` monorepo
this dataset ships from states blanket CC BY 4.0 for all its datasets.
**Correction applied**: license upgraded from "unclear" to "CC BY 4.0,
confirmed" — this was the repo being more conservative than the facts
supported, not the other way around. Exact record count also corrected:
**59,555** (from the paper's own abstract), not the previously-configured
approximate 60,000.

### 2.3 RICO — two independently-exported repos, two real bugs

**RICO core** (`creative-graphic-design/Rico`): **Bug found**: configured
to scan for loose `.jpg`/`.png`/`.json` files, but the live repo ships
*only* `.parquet` files across three named configs (`default` — metadata
only, no images; `ui-screenshots-and-view-hierarchies`; `ui-screenshots-
and-hierarchies-with-semantic-annotations`). The configured fetch would
have matched zero files and silently produced **zero training records**
from the single most foundational dataset in the UI-domain corpus (CLAY,
Enrico, and Screen2Words all join onto RICO records that would never have
existed). **Fix**: new `hf_parquet_images` fetch mode that decodes the
embedded image column directly; confirmed column name is `screenshot`.
**License**: dataset card's own metadata states `license: unknown`
explicitly — not "we haven't checked," the source itself never specified
one.

**RICO semantic** (`Voxel51/rico`): same failure class independently
confirmed (Voxel51 describes this repo as "Parquet formatted"). **Column
name confirmed directly from the live Data Studio preview table header**:
`image` — a *different* name from RICO core's `screenshot`, despite
covering the same underlying screens (two independently-exported
copies). **License**: dataset card states `License: CC BY 4.0` as a
top-level field — a firmer claim than the earlier "believed CC BY 4.0 for
the annotation layer" note.

### 2.4 HPDv2 (`ymhao/HPDv2`)

**Claim checked**: whether HPDv2 fits the generic fixed-two-image-column
preference-pair fetch path (the shape Pick-a-Pic v2 and the two
UI-domain datasets use).
**Finding**: HPDv2's real annotation format (`train.json`) is a
**variable-length ranked list per prompt** (`human_preference: list[int]`,
`file_path: list[str]`) — not a fixed two-column table, and not shipped
as parquet at all. The generic fetch path would have found zero
`*.parquet` files and returned nothing.
**Fix**: dedicated `hpdv2_ranked_list` adapter that flattens each
prompt's ranked comparison (2-candidate case only; ties/3+-way comparisons
are skipped and counted, not guessed at) into a winner/loser pair.
**License**: Apache 2.0, confirmed on the card. ~798K pairwise
comparisons / ~434K images.

### 2.5 Pick-a-Pic v2 (`yuvalkirstain/pickapic_v2`)

Confirmed real, MIT license, real human web-collected preference
comparisons (pickapic.io). Image bytes embedded directly in `jpg_0`/
`jpg_1` columns.

### 2.6 TASTE (`purvanshi/TASTE`) — held-out evaluation only

**Claim checked**: this was initially an educated guess at a repo name,
presented with appropriate hedging, not verified fact.
**Finding, on follow-up verification**: the paper describing it
(arXiv:2605.20731, Zhu et al., 2026) states directly in its own text:
*"Data: https://huggingface.co/datasets/purvanshi/TASTE"*, with analysis
code at `github.com/purvanshi-lica/taste` (MIT license per its README).
The guess was correct, but wasn't *known* correct until this check. 10
professional designers (two disjoint 5-person cohorts), 9 rating criteria
(typography, aesthetics, spatial, tone, ...), both pairwise comparisons
and derived 4-way rankings over 4 T2I models' outputs. Ships with its own
signal-validation framework (Kendall's tau, majority-vote probability,
Condorcet cycles against iid-uniform nulls).

### 2.7 PartiPrompts (`nateraw/parti-prompts`)

Confirmed real, Apache 2.0, ~1,632 text prompts only (no images), shipped
as a `.tsv` file — not parquet, correctly configured as such.

### 2.8 DesignSense-10k — confirmed absent, not just unconfirmed

**Claim checked**: does a public repo exist for this UI-domain DPO
dataset (arXiv:2602.23438, Adobe MDSR, Gopal et al., Feb 2026)?
**Finding**: no public dataset release found, checked twice (two
independent search passes, same result both times) directly against the
paper and its listing — no HF/GitHub data link anywhere.
**Additional finding, more consequential than the missing repo**: the
paper's own arXiv license badge confirms **CC BY-NC-ND 4.0** — Non-
Commercial *and* No-Derivatives. NoDerivs conflicts directly with this
project's recaption/re-annotation pipeline step; NonCommercial is a
separate concern for any product track. Even if a release appears later,
this needs dedicated legal sign-off, not just a repo-ID lookup.
Configured as `repo_id: null` in `data-forge/configs/datasets.yaml`,
tracked via the registry watcher for a future release.

### 2.9 DesignPref — confirmed *not yet released*, not a search failure

**Claim checked**: same question as above, for the second UI-domain DPO
candidate (Peng et al., 25 Nov 2025 paper).
**Finding**: this is not an unconfirmed gap — it's independently
corroborated by a *different* paper. TASTE's own related-work section
(arXiv:2605.20731) states outright: *"DesignPref describes a UI-design
pairwise dataset whose data has not yet been released."* There is no repo
to find because there isn't one yet. Same `repo_id: null` /
registry-watcher treatment as DesignSense-10k.

### 2.10 GameLabel-10K — found, deliberately not integrated

Surfaced during the DesignSense-10k/DesignPref re-check.
**Confirmed real**: `Jonathan-Zhou/GameLabel-10k`, Apache 2.0,
arXiv:2409.19830, ~10K human-labeled image-preference pairs from
mobile-game crowdsourcing, already demonstrated improving Flux-Schnell
via DPO/LoRA. **Not wired in**: ships as a single 2.26GB `data.csv`, not
parquet — doesn't fit either existing preference-pair fetch shape, and
its exact column names weren't independently confirmed against the live
file before this project's scope closed. Documented in
`docs/data-forge/DATA_SOURCES.md` as a vetted candidate for future
integration, specifically to avoid repeating the HPDv2/PD12M pattern of
wiring in a fetch path against a guessed schema.

---

## 3. Model stack decisions

### 3.1 Qwen-Image-2.0 → Qwen-Image-Edit-2511 (a correction, not a preference)

**Finding**: Qwen-Image-2.0 was announced Feb 10, 2026 as a chat-demo
launch; its weights were never released — confirmed via the QwenLM
GitHub changelog (demo link, not a weights link), an unanswered
HuggingFace community thread running Feb–Apr 2026 asking specifically
when weights would ship, and independent coverage stating outright the
weights weren't open-sourced. Alibaba had already moved on to announcing
Qwen-Image-3.0 without ever shipping 2.0's weights in between.
**Correction applied**: replaced with Qwen-Image-Edit-2511 (released Dec
23, 2025, confirmed downloadable, Apache 2.0, same MMDiT lineage) — this
was a build-blocking error caught before it could silently block the
entire Polish Quality tier at load time.

### 3.2 Gemma 4 31B Dense vs. 26B-A4B (MoE)

**Finding**: Unsloth has a documented, unresolved incompatibility between
the 26B-A4B variant's fused MoE expert-weight tensors and 4-bit
quantization — Unsloth explicitly does not ship bnb-4bit builds for it
and recommends 16-bit LoRA instead, which doesn't fit the VRAM budget.
The 31B Dense variant uses a different, non-MoE attention design
(interleaved local sliding-window + global attention) that quantizes
cleanly. **License**: Apache 2.0, confirmed genuine on both the Google
announcement and the HuggingFace LICENSE file — a real change from Gemma
1–3's more restrictive custom terms.

### 3.3 Qwen3.5-9B — training-time vs. inference-time quantization (a bug this project made and then caught)

**Original finding (§6.1)**: Qwen3.5's hybrid Gated DeltaNet + Gated
Attention mechanism has a documented QLoRA degradation problem during
**training**.
**The bug**: an earlier revision of `PlannerBackend` carried this
training-time finding into an inference-time decision — defaulting to
unquantized BF16 (~18GB for a 9B model) — even though the Planner ships
frozen under the no-RLHF-loop revision and never trains at all, making
the original finding moot for this tier. This conflicted directly with
`model_registry.py`'s declared `vram_gb=6.5` budget (a 4-bit-scale
number) for this `always_resident=True` tier, which would have blown the
PRD's "Tier A must never exceed ~10GB resident" rule before the sketch
tier or anything else could load.
**Fix**: `PlannerBackend` now defaults to NF4 (4-bit) inference, matching
the declared budget. This is explicitly *not* a claim that 4-bit
inference quality is validated for this model — that remains a genuinely
open question (see §5 below) — it's a claim that the budget and the
actual loading code now agree with each other, which they didn't before.

---

## 4. Low-VRAM / CPU-offload mode

Built after explicitly checking two real, documented compatibility
constraints — not assumed to "just work" because the individual pieces
(bitsandbytes, diffusers offload) are each independently well-known.

### 4.1 `enable_sequential_cpu_offload()` is NOT safe with bnb NF4

**Finding**: a real, documented GitHub issue (diffusers #10800) reports
`enable_sequential_cpu_offload()` combined with bitsandbytes 4-bit
quantization raising `"Blockwise quantization only supports 16/32-bit
floats, but got torch.uint8."` The less aggressive
`enable_model_cpu_offload()` is reported working fine in the same issue
thread.
**Decision**: `polish_quality_backend.py` (Qwen-Image-Edit-2511, NF4)
uses `enable_model_cpu_offload()` exclusively. `enable_sequential_cpu_
offload()` is never called anywhere in this codebase for an NF4-quantized
pipeline.

### 4.2 bitsandbytes' CPU-offloaded portion is stored in FP32, not 4-bit

**Finding**: `BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)`
is the real, documented flag required to combine `max_memory`-based CPU
offload with bnb quantization at all (omitting it raises a clear error:
*"Some modules are dispatched on the CPU or the disk..."*). The
documented behavior of this flag is that the CPU-resident portion is
stored in **FP32**, not 4-bit — an 8x per-parameter size increase
relative to NF4.
**Consequence, computed not guessed**: for Gemma 4 31B Dense (~30.7B
params), offloading enough to bring GPU residency down to ~12GB (roughly
a third of the ~18GB NF4 total) means ~10B params living on the CPU side
at FP32: **10B × 4 bytes ≈ 40GB of system RAM**. This is the real,
load-bearing number behind `LOW_VRAM_REGISTRY[Tier.CRITIC].ram_gb` in
`model_registry.py` — not a round-number placeholder.

### 4.3 Unsloth has no documented CPU-offload support

**Finding**: `FastModel.from_pretrained`'s documented API surface has no
`max_memory` or CPU-offload parameter; Unsloth's own issue tracker shows
users hitting plain OOM on VRAM-constrained cards rather than a graceful
offload path (e.g. a "Qwen3-30B-A3B not loading on RTX4090" report with
no offload workaround offered).
**Decision**: `critic_worker.py`'s low-VRAM path bypasses Unsloth
entirely and uses plain `transformers` + `bitsandbytes` instead (the same
verified pattern as §4.2) — slower than Unsloth's fast path, a real,
documented trade-off, not a bug.

### 4.4 Diffusion-pipeline offload does not have the FP32 problem

**Why Polish Quality's RAM estimate (~10GB) is so much smaller than
Critic's (~40GB) despite a similar VRAM target**: `enable_model_cpu_
offload()` moves whole submodules (text encoder, VAE, transformer)
between GPU/CPU, keeping their already-quantized dtype intact while idle
on CPU — no FP32 upcast. The RAM cost here is close to 1:1 with the VRAM
saved, unlike the bitsandbytes-LLM case in §4.2. This asymmetry is real,
not an inconsistency in the numbers.

**What's estimated vs. computed**: §4.2's ~40GB figure is a computed
lower bound from documented behavior. §4.4's ~10GB/~10GB split for Polish
Quality is a reasoned estimate, not a benchmark — validate against a real
run before treating it as exact (see `tests/inference/test_low_vram_mode.py`
for what's actually mechanically verified vs. what's a placeholder
number).

---

## 4.5 Diffusion-DPO for a flow-matching model — closing a previously-open gap

**The gap, as it stood**: `training/src/krisna_training/dpo/` built and
exported real preference-pair data (§2's human-labeled sources), but
nothing consumed it — there was no trainer.

**Why this isn't "just apply Diffusion-DPO"**: the original formulation
(Wallace et al., "Diffusion Model Alignment Using Direct Preference
Optimization", CVPR 2024, Eq. 46) is derived for DDPM-style models
predicting **noise** (epsilon). Z-Image-Turbo is a **flow-matching**
model predicting a **velocity** — a different prediction target with
different loss geometry. The DDPM-derived loss doesn't transfer
unmodified.

**The adaptation used, verified against real published precedent, not
invented for this project**: MotionFlux (Bin et al., arXiv:2508.19527,
§3.6) applies exactly the substitution needed — replacing the epsilon-
prediction squared-error terms in Wallace et al.'s sigmoid-of-difference
structure with velocity-prediction squared-error terms — to align a
rectified-flow-matching motion-generation model via DPO, citing Wallace
et al. as its own base formulation. The same paper also documents a real
failure mode (reward-hacking-style drift from the pretraining
distribution when optimizing DPO on a diffusion/flow model in isolation)
and addresses it with a flow-matching anchor regularization term — also
implemented here (`fm_anchor_weight`, off by default pending a real
sweep).

**What was verified vs. what needs a real GPU run**:
- **Verified, via genuine mathematical tests** (`tests/training/
  test_dpo_loss.py`), not just shape/smoke checks: the loss equals
  exactly `log(2)` when the policy hasn't diverged from the reference (the
  standard DPO closed-form sanity check — true for any correctly-
  implemented DPO-family loss), the loss correctly rewards the policy
  fitting the chosen sample better than the reference and correctly
  penalizes the opposite, a real gradient step measurably moves the
  chosen-side prediction toward its target, the anchor term's on/off
  behavior, and — specific to the flow-matching adaptation — that the
  velocity-target and noising-interpolant sign conventions are mutually
  consistent (predicting the target velocity and taking one Euler step
  from the noised latent recovers the original clean latent exactly).
- **Sign/target convention**: `target = noise - clean_latent`, matching
  diffusers' own `train_dreambooth_lora_sd3.py`/`train_dreambooth_lora_
  flux.py` scripts — deliberately not hand-derived from a paper's own
  notation (papers disagree on which endpoint is t=0 vs t=1; getting this
  backwards would be a silent, serious bug).
- **API surface used to wire the loop together** (`transformer.
  add_adapter(LoraConfig(...))`, `model.save_pretrained()` once PEFT-
  wrapped): confirmed real via direct research — a diffusers maintainer's
  own bug-report example calls `add_adapter()` on a real diffusers
  transformer model (github.com/huggingface/peft/issues/2494), and
  PEFT's own quickstart confirms `save_pretrained()` as the standard
  save call afterward. An earlier draft used an unconfirmed
  `save_lora_adapter()` method name that was never found documented
  anywhere in the same research pass — caught and replaced before it
  could ship as an untested guess.
- **Not verified, and can't be from this environment**: real training
  dynamics against actual Z-Image-Turbo weights, whether the default
  `beta=2000.0` (Wallace et al.'s own SD1.5/SDXL-tuned range) transfers
  reasonably to this model, and the two pipeline-specific API details
  flagged above.

---

## 5. Open questions this project deliberately leaves open

Documented here rather than silently resolved one way or the other:

- **4-bit inference quality for the Planner (Qwen3.5-9B) is unvalidated.**
  The training-time QLoRA degradation finding (§3.3) does not by itself
  say anything about inference-time NF4 quality — that's a different,
  separate, still-open question.
- **`KRISNA_POLISH_QUALITY_EDIT_STRENGTH`'s `strength` kwarg is unverified
  for `QwenImageEditPlusPipeline` specifically.** Confirmed real for
  `QwenImageImg2ImgPipeline`/`QwenImageInpaintPipeline` — architecturally
  different pipeline classes. If the installed diffusers version rejects
  this kwarg, it raises a clear `TypeError` at call time rather than
  silently no-op'ing; check the real `__call__` signature before relying
  on this in production.
- **Region-locking in `polish_quality_backend.py`** is currently expressed
  as a prompt-level instruction, not a genuine differential-diffusion
  mask parameter — diffusers' locked-region API for this pipeline family
  was still moving as of this build.
- **`train_dpo.py`'s `encode_prompt()` return shape and the transformer's
  forward-call signature are not independently verified against
  Z-Image-Turbo's specific pipeline class** — flagged inline in the code
  itself (same discipline as the `strength` kwarg above). The loss math
  driving the training loop is verified (§4.5 below); this specific
  pipeline-calling-convention detail is not, and needs a real
  `Tongyi-MAI/Z-Image-Turbo` pipeline inspection to nail down before a
  production run.

---

---

## 6. Sources verified in this document

| Source | What it verified |
|---|---|
| `huggingface.co/datasets/Spawning/PD12M` (dataset card) | Column schema, CDLA-Permissive-2.0 license |
| `github.com/google-research-datasets/clay`, paper copyright footer | CC BY 4.0 license, 59,555 exact record count |
| `huggingface.co/datasets/creative-graphic-design/Rico` (dataset card, Data Studio preview) | Parquet-only format, three named configs, `screenshot` column, "unknown" license |
| `huggingface.co/datasets/Voxel51/rico` (dataset card, Data Studio preview) | Parquet format, `image` column, CC BY 4.0 license |
| `huggingface.co/datasets/ymhao/HPDv2` (dataset card, `train.json` schema) | Ranked-list annotation format, Apache 2.0 |
| `huggingface.co/datasets/yuvalkirstain/pickapic_v2` (dataset card) | MIT license, embedded image columns |
| arXiv:2605.20731 (Zhu et al.) — TASTE paper | `purvanshi/TASTE` repo confirmation, MIT license, methodology |
| `huggingface.co/datasets/nateraw/parti-prompts` (dataset card) | Apache 2.0, `.tsv` format |
| arXiv:2602.23438 (Gopal et al.) — DesignSense-10k paper, its arXiv license badge | CC BY-NC-ND 4.0, no public data release |
| arXiv:2605.20731's related-work section | DesignPref confirmed not yet released |
| arXiv:2409.19830 — GameLabel-10K paper; `huggingface.co/datasets/Jonathan-Zhou/GameLabel-10k` | Real, Apache 2.0, unconfirmed schema |
| QwenLM GitHub changelog, HF community thread (Feb–Apr 2026) | Qwen-Image-2.0 weights never released |
| Unsloth documentation, Google Gemma 4 announcement, HF LICENSE file | Gemma 4 31B Dense vs. 26B-A4B MoE incompatibility, Apache 2.0 |
| `diffusers` GitHub issue #10800 | `enable_sequential_cpu_offload()` incompatible with bnb NF4; `enable_model_cpu_offload()` confirmed working |
| `transformers`/`bitsandbytes` documentation (`BitsAndBytesConfig`) | `llm_int8_enable_fp32_cpu_offload` requirement and FP32 CPU-storage behavior |
| Unsloth issue tracker | No documented CPU-offload support |
| Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization" (CVPR 2024) | Original Diffusion-DPO formulation (DDPM/noise-prediction), `beta` range |
| Bin et al., MotionFlux (arXiv:2508.19527) | Velocity-prediction DPO substitution for flow-matching models, flow-matching anchor regularization |
| `github.com/huggingface/peft/issues/2494`; PEFT quickstart docs | `transformer.add_adapter(LoraConfig(...))` and `model.save_pretrained()` confirmed real for diffusers models |
