# Phase 7 — Consolidated Citations & Open-Action Rollup

Merges every citation surfaced across Phases 1–6 with the repo's existing
`docs/architecture/RESEARCH_AND_CITATIONS.md`, and rolls up every open
action item from all six phases into one checklist. This is the
paper-ready reference list plus the punch list for finalizing the repo.

One minor doc-hygiene note on the existing file: its section numbering
has a duplication artifact — `## 4.5 Diffusion-DPO...` and a second
`## 5. Open questions...` both appear after the first `## 5.`, i.e. two
different `## 5` headers exist. Cosmetic only, but worth a quick
renumber pass before this doc goes into a paper's appendix.

## Consolidated bibliography (new citations from this review, by phase)

**Phase 1 — Sketch tier (MaskGIT)**
1. Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer*. CVPR 2022.
2. Chang, H., et al. (2023). *Muse: Text-to-Image Generation via Masked Generative Transformers*. ICML 2023. arXiv:2301.00704.
3. Besnier, V. & Chen, M. (2023). *A Pytorch Reproduction of Masked Generative Image Transformer*. arXiv:2310.14400. (Source for the ~387M-sample training-budget precedent cited against this repo's Stage-1 budget.)
4. Besnier, V., et al. (2025). *Halton scheduler as a plug-in replacement for confidence-based scheduling in masked generative transformers*. arXiv (2025).

**Phase 2 — Polish tier (Z-Image LoRA + Diffusion-DPO)**
5. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., & Naik, N. (2024). *Diffusion Model Alignment Using Direct Preference Optimization*. CVPR 2024.
6. (Corroborating source for β=2000, independent of the repo's own citation) arXiv:2410.05255 — confirms "β=2000, which stays the same in Diffusion-DPO."
7. Bin, Y., et al. (2025). *MotionFlux* [flow-matching DPO adaptation]. arXiv:2508.19527, §3.6.
8. Esser, P., et al. — Stable Diffusion 3 (logit-normal timestep sampling for flow matching).
9. Lumina-T2X team — arXiv:2405.05945 (logit-normal timestep sampling, corroborating source).
10. (2026) arXiv:2603.12517 — logit-normal sampling accelerates early convergence but caps final quality vs. a later-uniform curriculum; cited as an honest caveat, not a contradiction of the repo's choice.
11. Kirstain, Y., et al. — Pick-a-Pic v2 (`yuvalkirstain/pickapic_v2`, MIT).
12. Hao, Y., et al. — HPDv2 (`ymhao/HPDv2`, Apache 2.0).
13. DesignSense-10k — Adobe MDSR, arXiv:2602.23438 (CC BY-NC-ND 4.0; confirmed no public dataset release as of this review).
14. DesignPref — Peng et al. (25 Nov 2025); confirmed unreleased per TASTE, arXiv:2605.20731.

**Phase 3 — Planner (Qwen3.5-9B, RAG)**
15. Qwen Team, Alibaba (2026). Qwen3.5/Qwen3-Next hybrid architecture (Gated DeltaNet + Gated Attention) — cite the official technical report directly.
16. Yang, S., et al. — *Gated Delta Networks: Improving Mamba2 with the Delta Rule*.
17. Axolotl documentation — Qwen3.5 hybrid-architecture LoRA/QLoRA training notes (corroborating, not primary, evidence for the frozen-Planner design decision).
18. UICrit — `google-research-datasets/uicrit`, CC BY-ND.

**Phase 4 — Critic (Gemma 4 31B, frozen)**
19. ~~Gemma Team, Google DeepMind — Gemma model family (needs a specific Gemma-4 primary source before this goes in the paper; not independently verified in this review).~~ **Superseded by #26 below (Phase 12) — primary source now confirmed.**
20. Unsloth — VLM LoRA fine-tuning API documentation (relevant only if the deprecated Critic-training path is ever revived).

**Phase 5 — Data pipeline**
No new external citations; relies on the dataset citations already listed above (RICO/CLAY/Enrico/WebUI/Screen2Words/UICrit).

**Phase 6 — Inference orchestrator**
No new external citations; bitsandbytes' documented `llm_int8_enable_fp32_cpu_offload` behavior should be cited to the bitsandbytes docs directly if the RAM-offload arithmetic goes in the paper's appendix.

**Phase 8 fixes — synthetic-data generalization (`10_synthetic_data_generalization_fix.md`)**
21. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., Manassra, W., Dhariwal, P., Chu, C., Jiao, Y., & Ramesh, A. (2023). *Improving Image Generation with Better Captions* (DALL-E 3 technical report). OpenAI. — Source of the 95%/5% synthetic/ground-truth caption mixing ratio implemented in this pass.
22. Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598. — Source of the classifier-free-guidance conditioning-dropout mechanism (`cfg_dropout_prob=0.1`) implemented in this pass.
23. Wang, B., Li, G., Zhou, X., Chen, Z., Grossman, T., & Li, Y. (2021). *Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning*. UIST 2021. — Source of the 6.57-word average source-caption length that justifies `s05_recaption` existing at all.

**Phase 12 fixes — post-upgrade resync audit (`12_post_upgrade_resync_audit.md`)**
24. Qwen Team, Alibaba (2026-03-02). *Qwen3.5 Small Model Series (0.8B/2B/4B/9B) release notes.* Hugging Face / ModelScope. — Confirms Phase 3's Qwen3.5-9B identification and architecture claims (hybrid Gated DeltaNet + Gated Attention, native multimodal); no correction needed, cited as the dated primary release record.
25. Tongyi-MAI, Alibaba (2025). *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.* Technical report, `github.com/Tongyi-MAI/Z-Image/blob/main/Z_Image_Report.pdf`. — Primary source for Decoupled-DMD/DMDR distillation and the S3-DiT architecture underlying the Polish-Default tier; upgrades Phase 2's Z-Image-Turbo citation from secondary coverage to a verified primary report.
26. Google DeepMind (2026). *Gemma 4.* Official model page, `deepmind.google/models/gemma/gemma-4/`; documentation at `ai.google.dev/gemma/docs`. — Closes Phase 4's previously-unresolved citation gap (#19 above) for the Critic tier's base model.

 not re-derived here, but load-bearing for the paper:** PD12M (`Spawning/PD12M`), TASTE (`purvanshi/TASTE`), PartiPrompts (`nateraw/parti-prompts`), GameLabel-10K, and the full §3/§4 model-stack and low-VRAM-mode writeups — all independently verified in earlier project work per that doc's own account, cross-checked spot-fashion against this review's own independent findings (e.g. §3.3's Qwen3.5 NF4 bug matches Phase 3's independent finding; §4.5's Diffusion-DPO-for-flow-matching writeup matches Phase 2's independent finding) with **no contradictions found** between the two efforts.

## Rollup: every open action item, all phases

| Phase | Item | Status |
|---|---|---|
| 1 | Confirm `maskgit_model.py`'s default inference step count falls in the literature's 8–15 range | Open |
| 1 | Add Halton-ordering citation to `RESEARCH_AND_CITATIONS.md` | Open |
| 1 | Delete or properly implement `data-forge`'s dead `maskgit_vq` (Open-MAGVIT2) stage | Open |
| 1 | Add `label_smoothing=0.1` to the Sketch tier's cross-entropy loss | **Fixed and now genuinely tested** — `losses.py`; originally only syntax-checked (no `torch` available), later re-verified once `torch` was installed successfully (see item below) |
| 1 | Set explicit `AdamW(betas=(0.9, 0.96))` to match MaskGIT precedent | **Fixed and now genuinely tested** — `train.py`; same upgrade in verification status as above |
| 1 | State the Stage-1 training-budget shortfall (~1.6 epochs vs. precedent's ~300) explicitly in the paper | Open |
| 1 | Verify CLAY/Enrico/Screen2Words join actually resolves against `rico_semantic` when `rico_core` is excluded, and compute the real final usable-image count | Open (de-risked in Phase 5, not fully closed) — **still the top remaining item** |
| 2 | Delete or properly implement `data-forge`'s dead `s08_5_dpo_encoding.py` output | Open |
| 2 | Fix `sync_dpo_pairs.py`'s stale "no DPO trainer yet" docstring | **Fixed** — verified via direct read after edit |
| 2 | Pin an explicit LoRA rank for the base `polish_default` fine-tune; confirm compatibility with DPO's rank=32 | **Fixed** — `rank: 32` added, yaml-parse verified; the CLI-flag name itself (`--rank`) is not independently verified against the actual vendored diffusers script version (no network access to fetch `third_party/diffusers` here) |
| 2 | Confirm LoRA-on-distilled-Turbo-base serving assumption with a real qualitative check | Open |
| 3 | Find a primary source (or run a confirming ablation) for "Qwen3.5's hybrid architecture degrades under QLoRA" | Open |
| 4 | None — no active hyperparameters/training in this phase | N/A |
| 5 | Compute the exact final usable-image count from a real pipeline run (ties to Phase 1's RICO item) | Open |
| 5 | Run `pin_revisions.py` against live HF API before any real training run (mechanism verified present/tested, not yet exercised live) | Open |
| 6 | ~~Low-VRAM envelope math broken~~ — **retracted**, see Phase 6 doc's correction section: baseline is unloaded before every finalize/critique, so no sum-fit issue exists. Replaced by a low-severity item below. | Retracted (was a review error, not a repo bug) |
| 6 | Add a small explicit safety margin to the Critic's low-VRAM admission check — it fits its 12GB envelope with exactly zero headroom today | **Fixed** — `vram_safety_margin_gb` added to both ledgers + `SwapOrchestrator`, defaults to `0.0` (no behavior change), opt-in; verified against the real test suite (5/5 + 12/12 passing) |
| — | Renumber the duplicate `## 5` headers in `RESEARCH_AND_CITATIONS.md` | **Fixed** — also found and fixed a second bug in the process: the mislabeled section was a sources-verification table, not "open questions"; retitled `## 6. Sources verified in this document` and reordered so `## 4.5` sits before `## 5` in actual reading order |
| — | **Sandbox environment: `torch` was uninstallable** (disk space), blocking real verification of the Sketch tier fixes | **Fixed** — diagnosed as a stale partial install from an earlier failed attempt; cleaned up, freed 3.2GB, reinstalled cleanly from PyPI. All 246 tests in `tests/training/` + `tests/inference/` now pass for real. |
| 1/synthetic | Recaptioning's original `source_caption` was discarded before reaching training, making caption-style mixing impossible | **Fixed** — `s12_model_data_export.py` → `sync_sketch_tier.py` → `dataset.py`, 95/5 mix ratio per Betker et al. 2023 (DALL-E 3), tested against the real `SketchTokenDataset` code with a synthetic manifest (realized ratio 0.9489 vs. 0.95 target) |
| 1/synthetic | No classifier-free-guidance conditioning dropout existed anywhere in Sketch tier training | **Fixed** — `make_collate_fn`'s new `cfg_dropout_prob=0.1` (Ho & Salimans, 2022), formula-validated on synthetic embedding data (realized rate 0.1009 vs. 0.10 target) |
| 12 | `maskgit_vq` dead stage was silently causing `is_encoding_complete()` to be False for every `ui_first` record — empty Sketch-tier exports, full domain wrongly held out, on every run | **Fixed** — see `12_post_upgrade_resync_audit.md` §1; 114/114 `data_forge` tests passing |
| 12 | `vram_budget.py` docstring falsely claimed a nonzero VRAM safety-margin default existed | **Fixed** — comment corrected to match actual design; see §2 |
| 12 | `download_weights.py`'s local-artifact discovery paths were unverified guesses, wrong on every count | **Fixed** — see §3, verified against real training save code + dry-run test |
| 12 | Frontend read the wrong conversation-history field names | **Fixed** — see §4, verified via live integration test |
| 12 | `model_registry.py`'s `POLISH_DEFAULT` `vram_gb=8.0` looks under-sized for a 6B-param bf16 model against Z-Image-Turbo's own published VRAM guidance | **Open** — needs a real hardware measurement; see §5 |
| 12 | Phase 4's Gemma-4 citation gap | **Fixed** — primary source found, see §5 and citation #26 |
| 13 | `model_registry.py`'s `POLISH_DEFAULT` declared `quantization="NF4/NVFP4"` was factually false — backend never applied it, by design (train/inference precision match); `vram_gb=8.0` inherited that false assumption | **Fixed** — see `13_ram_offload_and_precision_audit.md`; quantization label corrected, `vram_gb` recomputed to 14.0 (arithmetic-grounded estimate, still pending real-hardware measurement), RAM-offload (`enable_model_cpu_offload()`) added and wired through `factory.py`, `LOW_VRAM_REGISTRY` entry added (12.0GB/2.0GB RAM) |

## Priority ordering, if working through this list top-down
1. **Phase 1's RICO join/count verification** — determines whether the paper's headline data-scale claim is accurate. Still the top open item; nothing fixed in this pass touches it.
2. **Confirm the sketch-tier hyperparameter fixes actually run** — they're syntax-checked and logically sound, but genuinely unverified against `torch` in this environment. Worth a real run before trusting them fully.
3. **Measure Polish-Default's real VRAM footprint on actual hardware** (Phase 12/13 finding) and confirm `model_registry.py`'s corrected `14.0`GB (`REGISTRY`) / `12.0GB+2.0GB RAM` (`LOW_VRAM_REGISTRY`) figures — these are now arithmetic-grounded, not wrong, but still not measured.
4. Everything else remaining is either a real-but-lower-severity fix (dead code paths, missing citations) or a citation/verification task that doesn't change what the system actually does.

---
This closes the six-phase review plan from `00_REVIEW_PLAN.md`. All seven docs (`00`–`06` plus this consolidated `07`) are in `docs/review/`. Phases 12–13 are later follow-up sessions' changelogs against this same set — see those files for anything after this line.
