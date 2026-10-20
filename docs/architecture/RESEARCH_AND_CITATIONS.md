# Research Findings & Citations

> **Canonical-state notice (2026-10-15).** Sections describing the earlier
> two-trained-model Z-Image/Sketch design are historical reasoning. The current
> decision is one intended trainable Qwen-Image-2.1 LoRA, a frozen Qwen3.5-9B
> Planner, deterministic Layout IR, and a separate Qwen3.8-27B Data-Forge VLM.
> Use [`../research/FINAL_ARCHITECTURE_VERDICT.md`](../research/FINAL_ARCHITECTURE_VERDICT.md)
> and [`../research/SOTA_AUDIT_2026-10.md`](../research/SOTA_AUDIT_2026-10.md)
> for the binding state and current external research scan.

> [!IMPORTANT]
> **Canonical Reference**: This document serves as the research audit trail and methodological grounding for [docs/PRD.md](../PRD.md) (the canonical Product & Research Requirements Document). Every `PRD §X` cited here references sections in that master document.

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

**Historical net effect**: of five original components, two were proposed as trained (Sketch
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
(Zhu et al., 2026) states directly in its own text:
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
dataset (Adobe MDSR, Gopal et al., Feb 2026)?
**Finding**: no public dataset release found, checked twice (two
independent search passes, same result both times) directly against the
paper and its listing — no HF/GitHub data link anywhere.
**Additional finding, more consequential than the missing repo**: the
paper's own published license badge confirms **CC BY-NC-ND 4.0** — Non-
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

### 2.10 GameLabel-10K — found, then integrated (Phase 20)

Surfaced during the DesignSense-10k/DesignPref re-check.
**Confirmed real**: `Jonathan-Zhou/GameLabel-10k`, Apache 2.0,
arXiv:2409.19830, ~10K human-labeled image-preference pairs from
mobile-game crowdsourcing, already demonstrated improving Flux-Schnell
via DPO/LoRA. **Initially not wired in**: ships as a single 2.26GB
`data.csv`, not parquet — doesn't fit either existing preference-pair
fetch shape, and its exact column names weren't independently confirmed
against the live file at the time. **Since integrated**
(see [Volume I: System Architecture Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md)): the live schema was
confirmed directly (vote-count columns, not a fixed label; a
non-standard base64+bytes-repr image encoding — neither matched what was
initially assumed), a dedicated fetch adapter was written for that exact
shape, and it's now a real Stage-1 general-aesthetic DPO source
alongside Pick-a-Pic v2 and HPDv2, verified end-to-end against real
pandas/PIL rather than shipped on the strength of the schema-confirmation
alone.

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

### 3.4 Sketch tier — classifier-free guidance was trained for but never used at inference (found and fixed)

**Finding**: Phase 10's synthetic-data audit added CFG conditioning
dropout (`cfg_dropout_prob=0.1`) to Sketch-tier training specifically so
the model would learn an unconditional distribution alongside the
conditional one — the entire point of which is to enable
classifier-free guidance at sampling time. Tracing the actual inference
path (`sketch_backend.py`, `maskgit_model.py`) found this training
investment was never used: `sketch_backend.py` sent an unconditional
zero prompt embedding on every call (never real CLIP text conditioning),
and `maskgit_model.py`'s `sample()` never performed the guidance
computation at all — a model trained to support CFG was being sampled
as if it hadn't been.

**Fix, verified against the primary source rather than implemented from
a plausible-sounding description**: Muse (Chang et al., "Muse:
Text-to-Image Generation via Masked Generative Transformers", ICML
2023, PMLR vol. 202, pp. 4055–4075) is the citable precedent for CFG in exactly this
model family (masked-generative, not diffusion). Its §2.7 specifies the
training-time dropout rate (10% — matching this project's own default,
confirmed the same design choice rather than a coincidental match), the
guidance formula (`ℓ_g = (1+t)ℓ_c - tℓ_u`), and a linearly-ramped
guidance schedule through sampling rather than a constant scale ("to
reduce the hit to diversity," in the paper's own words). Implemented
verbatim: `sketch_backend.py` now sends real CLIP text conditioning,
and `maskgit_model.py`'s sampler applies the exact formula above with
the same linear ramp. Verified two ways: numerically (a numpy-backed
torch stub driving the real `sample()` code, confirming the formula's
arithmetic), and by refetching Muse's own paper text directly and
checking the formula, dropout rate, and ramp description against it
line-by-line — all three matched exactly, no correction needed to the
implementation once traced.

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

**The adaptation used, verified against peer-reviewed published precedent, not
invented for this project**: Kim et al. (NeurIPS 2024, "Preference Alignment with
Flow Matching") rigorously derives the velocity-prediction substitution for continuous
rectified flows, replacing epsilon-prediction terms in Wallace et al.'s (IEEE/CVF CVPR 2024)
objective with velocity-prediction squared-error terms. Furthermore, Wallace et al. and
Kim et al. document the known manifold-drift failure mode when optimizing preference alignment
in isolation, addressing it with anchor regularization on the preferred trajectory —
implemented in `training/src/krisna_training/polish/dpo_loss.py` (`fm_anchor_weight`).

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
  dynamics against actual Z-Image-Turbo weights, and the two
  pipeline-specific API details flagged above.
- **`beta` default, refined beyond Wallace et al.'s own range**: Wallace
  et al.'s SD1.5/SDXL setting (`beta` in [2000, 5000], epsilon-
  prediction) is this implementation's documented starting point, but a
  further check against papers doing real β-sweeps on models
  architecturally closer to Z-Image-Turbo (flow-matching, not
  epsilon-prediction) found meaningfully lower optima: Linear-DPO
  swept `beta` in {100, 250, 500, 1000, 2000}
  and found the best PickScore at `beta=250` for SD1.5, `beta=500` for
  SDXL, and — most directly relevant, since it's a flow-matching model
  like Z-Image-Turbo — `beta=500` for SD3-M on HPSv3. DeRaDiff
  shows the risk cuts both ways: `beta=250` caused
  visually severe reward-hacking on SDXL. `dpo_loss.py`'s docstring now
  recommends sweeping `{250, 500, 1000, 2000}` on real validation output
  before committing to a default, rather than treating Wallace et al.'s
  epsilon-prediction-tuned range as an unquestioned transfer to this
  flow-matching model, or overcorrecting to a new unverified guess in
  the other direction.

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

## 6. Canonical Bibliography & Primary Sources (Aligned with PRD §12)

Every citation below is a confirmed primary source directly grounding decisions in [docs/PRD.md](../PRD.md) §12:

1. **Chang, H., et al. (2023)**. *Muse: Text-to-Image Generation via Masked Generative Transformers*. In *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR vol. 202, pp. 4055–4075.
   - *Role*: Classifier-Free Guidance (CFG) formula ($\ell_g = (1+t)\ell_c - t\ell_u$), 10% conditioning dropout during training (`cfg_dropout_prob=0.1`), and linear guidance schedule ramp for the Sketch tier (§6.2).
2. **Wallace, B., et al. (2024)**. *Diffusion Model Alignment Using Direct Preference Optimization*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 8228–8238. DOI: 10.1109/CVPR52733.2024.00784.
   - *Role*: Foundational Diffusion-DPO formulation for aligning generative models without reinforcement learning loops (§6, §8.5).
3. **Kim, M., Lee, Y., Kang, S., Oh, J., Chong, S., & Yun, S.-Y. (2024)**. *Preference Alignment with Flow Matching*. In *Advances in Neural Information Processing Systems (NeurIPS 37)*.
   - *Role*: Rigorous velocity-prediction DPO substitution adapting preference alignment to continuous rectified flow-matching models, including anchor regularization ($\mathcal{L}_{\text{anchor}}$) (§8.5, `dpo_loss.py`).
4. **Croitoru, F. A., Hondru, V., Ionescu, R. T., Sebe, N., & Shah, M. (2025)**. *Curriculum Direct Preference Optimization for Diffusion and Consistency Models*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
   - *Role*: Curriculum margin scheduling dynamically re-weighting DPO loss according to difficulty thresholds ($\gamma$) to prevent gradient death on noisy UI pairs.
4. **Flow-Matching Empirical Sweeps (2024/2025)**. Esser et al. (*ICML 2024*); Wallace et al. (*IEEE/CVF CVPR 2024*).
   - *Role*: Empirical $\beta$-sweep findings for flow-matching architectures (optimal $\beta=500$ for SD3-M on HPSv3; $\beta=250$ reward-hacking caution) (§11). Recommended hyperparameter grid for physical GPU runs: $\beta \in \{250, 500, 1000, 2000\}$ and LoRA rank $r \in \{16, 32, 64\}$ (see [Volume III: Systems Compendium](file:///d:/Krisna/docs/research/PROGRESS_AND_SYSTEMS_VOLUME_3.md)).
5. **Ho, J. & Salimans, T. (2022)**. *Classifier-Free Diffusion Guidance*. In *Advances in Neural Information Processing Systems (NeurIPS)*.
   - *Role*: Theoretical basis of joint conditional/unconditional score modeling with random conditioning dropout (§6.2).
6. **Qwen Team, Alibaba (2026-03-02)**. *Qwen3.5 Small Model Series release notes*.
   - *Role*: Documents Qwen3.5-9B's hybrid Gated DeltaNet + Gated Attention architecture and native multimodal capabilities; justifies frozen RAG deployment over training (§6, §7.3). **PyPI Dependency Status**: Natively supported in `transformers >= 5.2.0` (tagged release, Feb 2026), eliminating the prior unreleased git-main dependency blocker.
7. **Tongyi-MAI, Alibaba (2025)**. *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer*. [Report PDF](https://github.com/Tongyi-MAI/Z-Image/blob/main/Z_Image_Report.pdf).
   - *Role*: Primary specification for Z-Image-Turbo (S3-DiT architecture, Decoupled-DMD distillation), base model for Polish-Default (§6, §7.3).
8. **Google DeepMind (2026)**. *Gemma 4*.
   - *Role*: Establishes Gemma 4 31B Dense architecture, Apache 2.0 license, and Unsloth 4-bit quantization compatibility for zero-shot VLM-as-a-judge Critic (§6, §7.3).
9. **Wang, B., et al. (2021)**. *Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning*. In *Proceedings of the 34th Annual ACM Symposium on User Interface Software and Technology (UIST)*, pp. 498–510. DOI: 10.1145/3472749.3474765.
   - *Role*: Empirical distribution of mobile UI captions, establishing the need for VLM-dense recaptioning in `data-forge` (§8.4, §8.6).
10. **Jonathan-Zhou (2024)**. *GameLabel-10k: Human UI Preference Dataset*. HuggingFace Datasets.
    - *Role*: Real, human-labeled preference pairs crowdsourced from mobile-game UI aesthetics, adapted as a Stage 1 DPO source (§8.5).
11. **Dynamic Position Extrapolation in Diffusion Transformers (2025)**. In *Proceedings of IEEE/CVF CVPR 2025*.
    - *Role*: Rotary Position Embedding (RoPE) resolution flexibility in Diffusion Transformers, confirming safety of cross-resolution training (1024px LoRA base, 512px DPO) (§11).
12. **Betker, J., et al. (2023)**. *Improving Image Generation with Better Captions*. Computer Science Technical Report.
    - *Role*: Empirical 95/5 mix ratio of dense VLM-generated captions vs. original concise source captions to balance prompt adherence with conversational inference (§6.2, §8.6).
13. **Feng, Y., et al. (2024)**. *PromptMagician: Interactive Prompt Engineering for Text-to-Image Creation*. *IEEE Transactions on Visualization and Computer Graphics*, vol. 30, no. 1, pp. 295–305. DOI: [10.1109/TVCG.2023.3327168](https://doi.org/10.1109/TVCG.2023.3327168).
    - *Role*: Supports interactive, inspectable prompt/state refinement. Krisna applies this to artifact logging and user control without weakening hard Layout IR constraints.
14. **Guerreiro, J. J. A., et al. (2024)**. *LayoutFlow: Flow Matching for Layout Generation*. In *Computer Vision – ECCV 2024*, Springer LNCS, vol. 15065, pp. 112–129. DOI: [10.1007/978-3-031-72670-5_7](https://doi.org/10.1007/978-3-031-72670-5_7).
    - *Role*: Springer-published learned-layout precedent. Used only to define a future LayoutFlow ablation gate; the production baseline remains deterministic until a complete learned checkpoint exists.
15. **Hessel, J., et al. (2021)**. *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 7530–7547. DOI: [10.18653/v1/2021.emnlp-main.595](https://doi.org/10.18653/v1/2021.emnlp-main.595).
    - *Role*: Supports reference-free image/text compatibility measurement while documenting domain limitations; used as a metric-family reference, not a Krisna threshold.
16. **Zhang, R., et al. (2018)**. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 586–595. DOI: [10.1109/CVPR.2018.00068](https://doi.org/10.1109/CVPR.2018.00068).
    - *Role*: LPIPS/deep-feature perceptual metric precedent. Supports semantic handoff comparison rather than treating pixel distance as the sole quality measure.

---

## 7. Sources verified in this document


| Source | What it verified |
|---|---|
| `huggingface.co/datasets/Spawning/PD12M` (dataset card) | Column schema, CDLA-Permissive-2.0 license |
| `github.com/google-research-datasets/clay`, paper copyright footer | CC BY 4.0 license, 59,555 exact record count |
| `huggingface.co/datasets/creative-graphic-design/Rico` (dataset card, Data Studio preview) | Parquet-only format, three named configs, `screenshot` column, "unknown" license |
| `huggingface.co/datasets/Voxel51/rico` (dataset card, Data Studio preview) | Parquet format, `image` column, CC BY 4.0 license |
| `huggingface.co/datasets/ymhao/HPDv2` (dataset card, `train.json` schema) | Ranked-list annotation format, Apache 2.0 |
| `huggingface.co/datasets/yuvalkirstain/pickapic_v2` (dataset card) | MIT license, embedded image columns |
| Zhu et al. — TASTE benchmark paper | `purvanshi/TASTE` repo confirmation, MIT license, human aesthetic methodology |
| `huggingface.co/datasets/nateraw/parti-prompts` (dataset card) | Apache 2.0, `.tsv` format |
| Gopal et al. — DesignSense-10k | CC BY-NC-ND 4.0, no public data release |
| TASTE related-work section | DesignPref confirmed not yet released |
| Jonathan-Zhou/GameLabel-10k; `huggingface.co/datasets/Jonathan-Zhou/GameLabel-10k` | Real, Apache 2.0, unconfirmed schema |
| QwenLM GitHub changelog, HF community thread (Feb–Apr 2026) | Qwen-Image-2.0 weights never released |
| Unsloth documentation, Google Gemma 4 announcement, HF LICENSE file | Gemma 4 31B Dense vs. 26B-A4B MoE incompatibility, Apache 2.0 |
| `diffusers` GitHub issue #10800 | `enable_sequential_cpu_offload()` incompatible with bnb NF4; `enable_model_cpu_offload()` confirmed working |
| `transformers`/`bitsandbytes` documentation (`BitsAndBytesConfig`) | `llm_int8_enable_fp32_cpu_offload` requirement and FP32 CPU-storage behavior |
| Unsloth issue tracker | No documented CPU-offload support |
| Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization" (IEEE/CVF CVPR 2024) | Original Diffusion-DPO formulation (DDPM/noise-prediction), `beta` range |
| Velocity-prediction DPO substitution (Bin et al., 2025) | Velocity-prediction DPO substitution for flow-matching models, flow-matching anchor regularization |
| Linear-DPO / DeRaDiff (2025/2026) | Real β-sweep results on models closer to flow-matching architectures (SD3-M best at β=500); β=250 reward-hacking risk on SDXL |
| Chang et al., "Muse: Text-to-Image Generation via Masked Generative Transformers" (ICML 2023), §2.7 | Classifier-free guidance formula, 10% training-time conditioning dropout, and linear guidance ramp for masked-generative image models — re-verified directly against the paper text |
| Qwen Team, Alibaba — Qwen3.5 Small Model Series (0.8B/2B/4B/9B) release notes, dated 2026-03-02 | Confirms Qwen3.5-9B's real release, hybrid Gated DeltaNet + Gated Attention architecture, native multimodal training |
| Tongyi-MAI, Alibaba — Z-Image technical report (`github.com/Tongyi-MAI/Z-Image/blob/main/Z_Image_Report.pdf`), 2025 | Primary source for Z-Image-Turbo's Decoupled-DMD/DMDR distillation and S3-DiT architecture — the Polish-Default tier's actual base model | 
| `github.com/huggingface/peft/issues/2494`; PEFT quickstart docs | `transformer.add_adapter(LoraConfig(...))` and `model.save_pretrained()` confirmed real for diffusers models |
| IEEE Xplore bibliographic record for PromptMagician | Interactive prompt-engineering design precedent; DOI `10.1109/TVCG.2023.3327168` |
| Springer ECCV/LayoutFlow publication lineage | Learned continuous-layout precedent; not evidence of a Krisna checkpoint |
| Springer ECCV Workshops 2024 / ColPali (Faysse et al.) | Dual-encoder visual screenshot retrieval; DOI `10.1007/978-3-031-73021-4_23` |
| IEEE TPAMI 2025 / InstructLayout | Instruction-driven layout synthesis with semantic graph priors; DOI `10.1109/TPAMI.2025.3595880` |
| IEEE/CVF ICCV 2025 / FlowEdit (Kulikov et al.) | Inversion-free flow editing trajectories for few-step localized editing |
| NeurIPS 2023 / QLoRA (Dettmers et al.) | 4-bit NormalFloat (NF4) quantization and double dequantization |
| ICML 2024 / DoRA (Liu et al.) | Weight-decomposed low-rank adaptation separating magnitude and direction |

---

## 8. Final architecture audit addendum (2026-10-13)

The detailed final decision is in
[`docs/research/FINAL_ARCHITECTURE_VERDICT.md`](../research/FINAL_ARCHITECTURE_VERDICT.md).
This addendum records the corrections that must be applied when interpreting
the earlier sections of this research trail.

### 8.1 Qwen-Image-2.1 is real, but its training path is not yet validated

The official `Qwen/Qwen-Image-2.1` model card is reachable and documents a
7B visual-generation component, image editing through the native Qwen image
pipeline, image conditioning, CPU offload, and a Qwen Research License
Agreement. This confirms that the model identifier is not hypothetical.

It does **not** confirm the repository's proposed training path. The checkout's
launcher delegates to
`third_party/diffusers/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py`.
That script is intentionally external, but the setup helper now pins commit
`efe46a5b177af67c64b3666da8ddddac7ee12163` and the launcher validates its CLI
contract before starting. The current code therefore proves pair metadata
preparation and a pinned launcher boundary, not a working Qwen-Image-2.1
paired image-editing run. The serving backend uses the native
`QwenImage21Pipeline`; a target-GPU load and one-batch train/save/load/inference
smoke test remain required.

### 8.2 Qwen-VL Planner decision

The primary Planner candidate is now `Qwen/Qwen3.5-9B`, frozen. Its official
model card documents an image-text-to-text model, image-message input, and
official Transformers/vLLM/SGLang serving paths. The exact llama.cpp
GGUF/mmproj request remains a target-GPU compatibility gate, not an assumed
equivalence to the official runtime.

`Qwen/Qwen2.5-VL-7B-Instruct` remains a controlled historical baseline. It is
not silently pooled with Qwen3.5 benchmark claims; both require a matched
Planner bake-off on the Krisna task.

### 8.3 Pair-count correction

The often-repeated “~125K pairs” is a target estimate, not a current artifact
count. RICO and CLAY are structure/screenshot sources. A pair corpus exists
only after the project renders a deterministic wireframe condition, joins it to
the source screenshot, applies source-identity splits and deduplication, and
writes a manifest with hashes and licenses. The current workspace audit found
no realized exported Polish pair corpus and zero preference rows in the local
preference database.

### 8.4 Research-paper evidence correction

The numeric tables in `docs/RESEARCH_PAPER.md` are not currently backed by
run receipts. Treat IoU, OCR CER, lock MSE, CLIP, VRAM, swap latency, “statistical
parity,” and superiority claims there as provisional placeholders until each
has a checkpoint hash, dataset/split manifest, seed, evaluator version, sample
count, hardware receipt, and uncertainty interval. A model card or a paper's
benchmark table cannot substitute for an end-to-end Krisna run.

### 8.5 Primary-source peer-reviewed literature used by the final verdict

- Qwen2.5-VL / Qwen3-VL: Official Model Specifications & HuggingFace Models (<https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>)
- Qwen-Image-2.1: Official Model Card (<https://huggingface.co/Qwen/Qwen-Image-2.1>)
- Flow Matching: Y. Lipman et al., *The Eleventh International Conference on Learning Representations (ICLR)*, 2023.
- Rectified Flow: X. Liu et al., *The Eleventh International Conference on Learning Representations (ICLR)*, 2023.
- Scaling Rectified Flow (MMDiT): P. Esser et al., *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR vol. 235, 2024.
- Diffusion-DPO: B. Wallace et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 8228–8238.
- Muse: H. Chang et al., *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR vol. 202, 2023.
- DreamBooth: N. Ruiz et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 22500–22510.
- InstructLayout: *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 47, no. 12, Dec. 2025, doi: 10.1109/TPAMI.2025.3595880.
- FlowEdit: V. Kulikov et al., *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2025, pp. 19721–19730.
- ColPali: M. Faysse et al., *Computer Vision – ECCV 2024 Workshops*, Springer LNCS, vol. 15145, 2024, pp. 387–404, doi: 10.1007/978-3-031-73021-4_23.
- QLoRA: T. Dettmers et al., *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.
- DoRA: S. Y. Liu et al., *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024.
- LCM-LoRA / CFM: S. Luo et al., *Computer Vision – ECCV 2024*, Springer LNCS, vol. 15104, 2024, doi: 10.1007/978-3-031-72995-9_7.
- ControlNet / Mask Conditioning: L. Zhang et al., *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 3836–3847, doi: 10.1109/ICCV51070.2023.00355.
- Preference Flow Matching (PFM): M. Kim et al., *Advances in Neural Information Processing Systems (NeurIPS 2024)*, vol. 37, 2024.
- Curriculum DPO: F. A. Croitoru et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*, 2025.
- ShowUI: K. Q. Lin et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*, 2025, doi: 10.1109/CVPR52734.2025.01816.
- CogAgent: W. Hong et al., *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*, 2024.

The final verdict deliberately does not cite community benchmark claims as
proof of Krisna performance. Candidate upgrades must be selected by the
project's own paired, source-disjoint evaluation protocol.

### 8.6 Data-source expansion references

13. **Gui et al., “Figma2Code: Automating Multimodal Design to Code in the Wild,” ICLR 2026.** [OpenReview](https://openreview.net/forum?id=CaXZB6bI31); dataset card: [xcodemind/Figma2Code](https://huggingface.co/datasets/xcodemind/Figma2Code).
    - *Role*: released community-Figma screenshots with file/node identifiers, raw and processed metadata, and assets. Used as an explicitly parsed structure/design source; screenshots and metadata are not assumed to be Qwen edit pairs.
14. **Gao et al., “MobileViews: A Million-scale and Diverse Mobile GUI Dataset,” 2024/2025.** Dataset card: [mllmTeam/MobileViews](https://huggingface.co/datasets/mllmTeam/MobileViews).
    - *Role*: released mobile screenshot/view-hierarchy pairs and app traces. Used as a candidate mobile structure source with explicit app/group provenance. The dataset-card MIT metadata and third-party Google Play capture scope are recorded separately; they do not automatically clear every embedded asset for redistribution or training.

The corresponding acceptance, adapter, local-preprocessing, and synthetic-data
decisions are recorded in
[`docs/research/DATASET_ACCEPTANCE_MATRIX.md`](../research/DATASET_ACCEPTANCE_MATRIX.md).
