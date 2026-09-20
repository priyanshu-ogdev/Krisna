# Phase 1 — Sketch Tier (from-scratch MaskGIT-style transformer)

> **RESOLVED (post-Phase-1 follow-up session):** the `maskgit_vq` dead-stage
> finding below (line ~26) has been acted on — option (a) from this doc's
> own recommendation. `data-forge/configs/models.yaml`'s `maskgit_vq`
> entry, `engine.py`'s always-failing loader, and `s08_encoding.py`'s dead
> encode branch have all been deleted; `utils/completeness.py` no longer
> requires `vq_tokens` for `ui_first` records. This also surfaced and
> fixed a **more severe, previously-undetected bug this doc didn't catch**:
> because the dead branch's failure was silently swallowed, `vq_tokens`
> was never present on any record, which made `is_encoding_complete()`
> False for every `ui_first` record — meaning `s09_heldout.py` excluded
> the entire `ui_first` domain from every run, and
> `s12_model_data_export.py` exported an **empty Sketch tier on every
> run**. See `docs/review/12_post_upgrade_resync_audit.md` §1 for the
> full account and verification (114 `data_forge` tests passing,
> including new tests for the corrected completeness behavior).

## What it is
A bidirectional (non-causal, BERT/ViT-style) transformer trained from
scratch over VQGAN token grids, with a learned MASK token, prefix-token
text conditioning, and a Token-Critic head. Progressive two-stage
training: Stage 1 at 16×16 tokens (256px), Stage 2 at 32×32 tokens
(512px), Stage 2 initialized from Stage 1 with bicubically-interpolated
positional embeddings.

- Config: `training/configs/sketch_train_stage{1,2}_*.yaml`
- Model: `training/src/krisna_training/sketch/model.py`
- Training-time masking: `training/src/krisna_training/sketch/masking.py`
- Inference sampler: `inference/src/krisna_inference/backends/maskgit_model.py`, `sketch_backend.py`

## Data → Training sync
- Tokenizer/codebook: `vocab_size: 16384` in both stage configs, documented as matching `boris/vqgan_f16_16384`'s codebook. Consistent across `model.py`'s `mask_token_id = vocab_size` (one past the codebook) and the dataset prep path — **in sync**.
- Stage 1→2 handoff: `init_from` in stage-2 config points at stage-1's final checkpoint; `pos_embed.py` is explicitly responsible for bicubic-interpolating 16×16→32×32 positional embeddings on load. This is architecturally the correct way to do progressive-resolution transfer for a learned (non-sinusoidal) position embedding — **in sync, and matches precedent** (progressive resizing is standard in ViT/MaskGIT-family training, e.g. FixRes/ViT progressive-resolution fine-tuning).
- `critic_loss_weight: 0.5` in both stages — a secondary Token-Critic loss alongside the primary masked-token cross-entropy, matching the Token-Critic mechanism described in the MaskGIT lineage (self-supervised confidence estimation on top of token prediction) rather than being an unexplained extra loss term.

## Data-forge ↔ Training sync (corrected — the original pass above only checked training-internal consistency, not the full path from data-forge)

**Not in sync at the intended design level, but not silently broken —
already flagged, worked around, and documented in-repo.**

- `data-forge/configs/models.yaml`'s `maskgit_vq` encoder = `TencentARC/Open-MAGVIT2`. Sketch training's actual tokenizer (`training/sketch/vq_tokenizer.py`, `model.py`, both stage configs) = `boris/vqgan_f16_16384`. Different codebooks, incompatible token spaces.
- `data-forge/data_forge/inference/engine.py`'s `maskgit_vq` loader **always raises `RuntimeError` by design** — Open-MAGVIT2 isn't a transformers-native repo, and the loader refuses to guess at an unverified `AutoModel.from_pretrained(..., trust_remote_code=True)` load. So `s08_encoding.py`'s "Sketch Tier VQ Tokens" branch (which runs once per `ui_first` record) fails every time, caught by a local `except Exception: log.warning(...)` — it never produces a usable `vq_tokens/*.pt`.
- Even if it did, `training/data_forge_bridge/sync_sketch_tier.py` explicitly does **not** consume `vq_tokens/*.pt` — it re-tokenizes data-forge's raw `images/` (linked into the export specifically for this) through the real, working `boris/vqgan_f16_16384` tokenizer.
- Net effect: the real training path works end-to-end (real images → real matching tokenizer → real manifest → real Sketch training), but data-forge's own `maskgit_vq` stage is dead weight — a guaranteed-fail encode attempt (plus log noise) per `ui_first` record, for an artifact nothing downstream reads.

**Action, not yet done:** either (a) delete `s08_encoding.py`'s Branch 2 and the `maskgit_vq` entry in `models.yaml` since it does nothing today, or (b) build the real Open-MAGVIT2 `.encode()`/`.decode()` wrapper and switch Sketch training onto it, dropping the bridge's re-tokenize step. (a) is the lower-risk fix; (b) is only worth it if Open-MAGVIT2's codebook is actually preferred over `boris/vqgan_f16_16384` for this use case — not established anywhere in the repo's own docs.

## Training ↔ Inference sync
- **Verified, not a bug**: `SketchModelConfig.prompt_dim` defaults to `4096` in `model.py`, while `sketch_train_stage*.yaml` (via `train.py`'s own `TrainConfig.prompt_dim: int = 768`, "CLIP ViT-L/14 text-embedding dim") overrides it to `768` for actual training runs. At inference, `sketch_backend.py` reads `prompt_dim` back off the *loaded checkpoint's own config* (`getattr(self._model.module.cfg, "prompt_dim", 4096)`) rather than assuming a fixed value, so the 4096 in both places is only a shared fallback default, not a live mismatch. This is a legitimately correct pattern (config travels with the checkpoint) — worth keeping, but the inline comments in `model.py`/`sketch_backend.py` are confusing enough (each says it "matches" the other's placeholder) that a future edit could break this silently. **Recommendation: add an explicit assertion at checkpoint-load time that `prompt_dim` matches the text embedder actually in use, rather than relying on comment-level agreement.**
- Training masking (`masking.py`) is explicitly documented as intentionally different from inference sampling (`maskgit_model.py`'s Halton-ordered, confidence-gated multi-round reveal) — this is correct MaskGIT practice, not a sync gap: train-time uses random-ratio/random-position masking so the model learns to fill any mask configuration; inference uses a structured reveal schedule. Flagging as **confirmed correct-by-design**, since a naive reviewer could mistake this for a training/inference mismatch.

## Hyperparameters vs. precedent — checked directly against Chang et al. 2022's Experimental Setup (§4.1), not from memory

**Matches the paper:** `n_heads=8`, `dropout=0.1` — exact. `ffn_dim/hidden_dim` ratio is 4× in both (paper: 3072/768; repo: 2048/512), internally proportional despite the repo's deliberately smaller scale. Progressive 256→512 resolution transfer mirrors the paper's own reuse-across-resolutions finding.

**Two concrete, fixable gaps:**
1. **No label smoothing.** `losses.py`'s `F.cross_entropy` call passes no `label_smoothing` (PyTorch default 0.0). The paper uses `label_smoothing=0.1`. Real, low-cost training-quality gap, not a deliberate scale trade-off.
2. **AdamW betas are framework defaults, not the paper's.** `train.py` calls `AdamW(..., lr=cfg.lr, weight_decay=cfg.weight_decay)` with no explicit `betas=`, silently getting `(0.9, 0.999)`. The paper uses `Adam(β1=0.9, β2=0.96)` — a deliberately lower second-moment decay, a known stabilization choice for masked-token transformer training, not an arbitrary pick. Should be set explicitly.

**Real convergence-budget concern, currently undisclosed:** Stage 1 = `100,000 steps × batch 32` ≈ 3.2M samples seen ≈ **~1.6 passes** over the stated ~2M-image target corpus. The original paper trains 300 epochs on ImageNet (1.28M images/epoch) ≈ 384M samples; the standard PyTorch reproduction (Besnier & Chen, arXiv:2310.14400) uses 755,200 iterations × batch 512 ≈ **387M samples** — roughly **two orders of magnitude** more gradient-update exposure than this repo's Stage 1 budget. Unlike the DPO config, which explicitly flags its hyperparameters as untuned starting points, the Sketch tier configs carry **no equivalent disclaimer** — this should be stated plainly in the paper as an intentional compute-constrained scope decision, not left implicit, since by every public precedent for this architecture the current budget alone would leave the model meaningfully undertrained.

## Diffusion convergence note
Not applicable to this phase — the Sketch tier is a masked-token transformer, not a diffusion model. See `02_polish_tier.md` for the diffusion (Z-Image-Turbo DPO) convergence-hyperparameter review.


## Inference strategy vs. SOTA
The repo's chosen inference-time approach — masked-token init, per-round confidence-based reveal, cosine mask schedule, Token-Critic-gated confidence — is the core MaskGIT decoding algorithm (Chang et al., 2022), which remains the right choice for this model family: it is specifically what makes MaskGIT ~8–64× faster than autoregressive token decoding at comparable or better FID (Chang et al., 2022; Synced/Google Research summary, 2022).

Two concrete, citable opportunities the repo does **not** yet mention, worth adding before finalizing this phase for the paper:
1. **Halton-sequence token ordering** (already used per the docstring reference to "Halton-ordered... reveal") is validated in recent work as a plug-in replacement for purely confidence-based reveal ordering that improves FID/IS/recall without retraining (Besnier et al., 2025) — the repo already does this; it should be explicitly cited in `RESEARCH_AND_CITATIONS.md` since it's a specific, checkable design decision, not generic MaskGIT.
2. **Step-count sweet spot**: the literature reports 8–15 decoding steps as the quality/diversity optimum at 256–512px for MaskGIT-family models, with excess steps *reducing* diversity (Chang et al., 2022; Besnier et al., 2023). Recommend the repo's inference config expose and default `num_steps` inside this documented range if it does not already — need to check `maskgit_model.py`'s default step count explicitly in the next pass (not yet confirmed against this range).

## Real data quantity & quality (genuine, non-synthetic sources only)

Data-forge's own claim (`DATA_COMPLETENESS.md`): **"~410K real UI images
across RICO/CLAY/Enrico/WebUI."** Verified per-source, against
`datasets.yaml`'s confirmed counts:

| Source | Count | License | Usable now? |
|---|---|---|---|
| RICO (core) | 66,261 | `unknown` — dataset card never specifies one | **No** — routes to `excluded_pending_review`, needs manual sign-off |
| RICO (semantic) | 66,261 (same screens, separate export) | CC BY 4.0, confirmed | Yes |
| CLAY | 59,555 (paper-confirmed) | CC BY 4.0, confirmed | Yes |
| Enrico | 1,460 | Academic/unverified | Marginal — tiny, mostly held-out eval not training volume |
| WebUI | 350,000 | Per-source COPYRIGHT.txt, unverified | Provisional |

**Real gap, not yet resolved in this pass:** CLAY, Enrico, and
Screen2Words are all annotations/captions *joined onto RICO's images*,
not standalone image sets. If `rico_core` is excluded pending manual
license review (which the license-agent's own routing logic says it
should be), it's not confirmed in this pass whether those joins
fail over to `rico_semantic` (same 66,261 screens, confirmed CC BY 4.0,
different HF export/column name) or simply lose ~66K images and
everything joined on top of them. This is the single biggest open item
for finalizing Phase 1's data-quantity claim — the "~410K" figure may be
overstated by tens of thousands of images plus broken joins if no
failover exists. **Action: trace the join code directly against
`rico_semantic` before citing a final usable-image count in the paper.**

Quality controls genuinely in place (not just claimed): safety/NSFW
classification, PII scrubbing (image + text), cross-source dedup, and
`ui_first_ratio` domain-balance enforcement are all real pipeline
stages with tests, not aspirational. The known per-chunk (not
corpus-wide) ratio-enforcement limitation from `DATA_SOURCES.md` still
stands as a minor, documented caveat.

## Findings summary

| Severity | Finding |
|---|---|
| Info | `prompt_dim` dual-default pattern is correct but fragile — add a load-time assertion (see above) |
| Info | Training-vs-inference masking difference is by design, not a bug — documenting explicitly to prevent future false-positive "fixes" |
| Action | Confirm `maskgit_model.py`'s default inference step count falls in the empirically-supported 8–15 range; cite if so, flag if not |
| Action | Add explicit citation for Halton-ordering choice to `RESEARCH_AND_CITATIONS.md` (currently implemented but uncited) |
| **Real gap** | `data-forge`'s `maskgit_vq` (Open-MAGVIT2) encoding stage is dead — always fails at load, output never consumed anyway (bridge re-tokenizes raw images through the real tokenizer instead). Functionally worked around, not fixed. Recommend deleting the dead stage or building the real wrapper — see above. |
| **Real gap** | No `label_smoothing=0.1` (paper's value); `AdamW` betas left at framework default `(0.9,0.999)` instead of the paper's `(0.9,0.96)`. Both cheap, concrete fixes. |
| **Real gap, undisclosed** | Stage 1's training budget (~1.6 effective epochs over target corpus) is ~2 orders of magnitude below both the original paper's and the standard PyTorch reproduction's sample-exposure — should be explicitly flagged as a compute-constrained scope decision in the paper, not left implicit. |

## Citations (Phase 1)

1. Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer*. CVPR 2022.
2. Chang, H., Zhang, H., Barber, J., Maschinot, A. J., Lezama, J., Jiang, L., Yang, M.-H., Murphy, K., Freeman, W. T., Rubinstein, M., et al. (2023). *Muse: Text-to-Image Generation via Masked Generative Transformers*. ICML 2023. arXiv:2301.00704.
3. Besnier, V., et al. (2023). *Halton Scheduler for Masked Generative Image Transformer*. (vanilla/PyTorch MaskGIT reproduction + scheduler study.)
4. Besnier, V., et al. (2025, Mar 21). *Halton scheduler as plug-in replacement for confidence scheduling in masked generative transformers* — improves FID/IS/recall without retraining. arXiv (2025).
5. Google Research team via Synced (2022, Feb 14). *Google's MaskGIT Outperforms SOTA Transformer Models on Conditional Image Generation and Accelerates Autoregressive Decoding by up to 64x*. (secondary summary source — cite primary Chang et al. 2022 in the paper itself, this only for context.)

---
Next: Phase 2 — Polish tier (Z-Image-Turbo, LoRA + Diffusion-DPO, two-stage general→UI-domain preference alignment).
