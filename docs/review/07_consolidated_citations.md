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
2. Chang, H., et al. (2023). *Muse: Text-to-Image Generation via Masked Generative Transformers*. ICML 2023. arXiv:2301.00704. — Re-verified in `21_frontend_and_scripts_merge.md` §5 directly against the paper (§2.7, equation 1: `ℓ_g=(1+t)ℓ_c-tℓ_u`; 10% training-time dropout; linear guidance ramp) rather than trusted from Phase 17's own citation — confirmed exact on every specific.
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
| 15 | `s08_5_dpo_encoding.py` still ran by default (`enabled: true`) despite `02_polish_tier.md` having already confirmed, in an earlier phase, that its output has zero real consumers (same orphaned-producer shape as `maskgit_vq`) | **Fixed** — see `15_data_forge_finalization.md` §4; disabled by default in `pipeline.yaml`, wire-vs-delete decision left explicit rather than silently resolved |
| 16 | **Highest-severity finding in this review**: `s04_5_escalation` ran AFTER, not before, `s05_recaption`/`s06_structure` in `orchestrator.py`'s actual `execute_pipeline()` order — any record Tier-2 escalation rescued (borderline→safe) was permanently, silently dropped from the training pool, never recaptioned/structured/routed/encoded. Not catchable by pairwise stage-filter checks (every individual filter was correct); only visible by reading the real multi-phase execution order | **Fixed** — see `16_preprocessing_ordering_audit.md`; phase order corrected (escalation moved before recaption/structure), regression test added asserting real call order, 241/241 tests passing |
| 17 | Sketch-tier inference never used either training-side fix from Phase 10: `sketch_backend.py` sent an unconditional zero prompt embedding on every call, and `maskgit_model.py`'s sampler never performed classifier-free guidance despite CFG-dropout training specifically enabling it | **Fixed** — see `17_sketch_inference_conditioning_and_cfg.md`; real CLIP text conditioning wired, CFG implemented per Muse (Chang et al. 2023) with the real `l_g=(1+t)l_c-t*l_u` formula, verified numerically and via a numpy-backed torch stub driving the real `sample()` code |
| 18 | Training-memory audit against the real A6000 48GB/128GB-RAM target: Sketch Stage 2 and Planner had no gradient checkpointing (Planner's activation memory upper-bound computed at ~245GB vs. ~7.7GB with checkpointing at its configured batch/seq length); DPO had no LR scheduler, loaded a fully redundant reference-pipeline VAE/text-encoder copy, and held both transformer copies on GPU simultaneously | **Fixed** — see `18_training_memory_audit.md`; also documents a self-correction — an earlier "standardize LoRA alpha to 2x across tiers" fix was reverted after research showed Critic/DPO each follow a real, different, model-API-specific convention than the assumption that motivated it |
| 19 | **Second-highest-severity finding in this review**: `service.py` never wired the real VQ-token-decode hook into Finalize — every real `/finalize` call crashed with `PIL.UnidentifiedImageError`, not a hypothetical. Also: raw JSON leaking into every planner chat message (the JSON delta's span was computed but discarded), and a critique-results panel fetched/stored but never rendered | **Fixed** — see `19_agentic_workflow_io_audit.md`; handoff hook wired with a clear-failure fallback, JSON stripped from user-facing text (caught and fixed a bug in the first attempt at this fix via direct execution), critique panel added to the frontend |
| 20 | GameLabel-10K listed as "vetted but not integrated" — real dataset, but neither existing fetch mode matches its actual schema (vote-count columns, not a fixed label; a non-standard base64+bytes-repr image encoding) | **Fixed** — see `20_gamelabel_10k_integration.md`; dedicated fetch adapter added after confirming the live schema, verified end-to-end against real pandas/PIL with two bugs caught before shipping |
| 21 | A restored regression test (`test_critic_low_vram_tier_has_real_headroom_not_exact_equality`) re-asserted the exact misconception Phase 6 already retracted — baseline+candidate VRAM summed together, rather than the real per-tier-alone admission model | **Fixed** — see `21_frontend_and_scripts_merge.md` §2; incorrect final assertion removed with an explanatory comment |
| 21 | A separate test (`test_low_vram_env_var_sets_critic_max_gpu_gb`) still asserted the pre-fix `max_gpu_gb == 12.0` instead of the corrected `11.5` | **Fixed** — see §2 |
| 21 | `data-forge/orchestrator.py`'s module docstring + duplicate "Phase 5" numbering (fixed once already) had reverted again in a later uploaded working copy | **Fixed again** — see §3 |
| 21 | `setup.sh` had no phase for `inference-frontend/` — no way for a fresh operator to discover or install the Node control panel | **Fixed** — see §4; new `frontend` phase, not in the default list, plus a next-steps mention |
| 22 | `scripts/inference/download_weights.py` (the real inference installer) had zero awareness of the VQGAN decoder `service.py` requires for every real Finalize call — an operator following the documented setup flow would hit a guaranteed runtime crash with no prior signal | **Fixed** — see `22_root_orchestration_and_vqgan_gap.md` §2; auto-discovery + auto-download added, verified via dry-run |
| 22 | Three shell-script "next step" hints (Sketch stage 1/2, VQGAN download), claimed added in a prior session, were not actually present on disk | **Fixed again** — see §3 |
| 22 | `tests/training/test_dpo_loss.py`'s bare `import torch` broke collection for the ENTIRE `pytest tests/` run when torch is absent, contradicting the root README's own "skips cleanly" claim | **Fixed** — see §4; matches the established `importorskip` pattern, 356 tests now collect and run cleanly (346 pass, 10 skip) |
| 22 | Root README and `src/README.md` both stale — no mention of the root scripts or `inference-frontend/`; `src/README.md` listed API endpoints that never existed | **Fixed** — see §5; both rewritten against verified current code |
| 23 | `inference/README.md`'s VRAM/RAM table and three prose mentions described pre-fix Polish-Default (8.0GB) and Critic (~12GB/~40GB) figures as current, found in a prior session's own follow-up pass | **Confirmed fixed** — see `23_s08_5_stale_doc_sweep.md` §1 |
| 23 | `s08_5_dpo_encoding` described as active/load-bearing across `data-forge/README.md` and three `docs/data-forge/*.md` files despite being disabled since Phase 15 — the same doc-drift class, left as "a known cleanup item" by a prior session under time pressure | **Fixed** — see §2; every table, flow diagram, and prose mention corrected against the real `sync_dpo_pairs.py` → `train_dpo.py` path |
| 24 | `sketch_backend.py`'s CFG shape-mismatch-guard fallback and `model.py`'s `prompt_dim` dataclass default both used a stale placeholder (4096) instead of CLIP ViT-L/14's real 768-dim output — dormant on any real checkpoint (which correctly carries 768), but a landmine for a legacy/malformed one | **Fixed** — see `24_tensor_shape_final_check.md`; both defaults corrected to 768 |
| 25 | Polish's two training stages disagree on resolution with zero acknowledgment: the base LoRA fine-tune trains at 1024, but the DPO refinement stage that continues the *same* adapter trains at 512 — found via pairwise-diffing config values across stage boundaries, not visible from either file alone | **Researched and documented, not silently changed** — see `25_polish_resolution_cross_stage_check.md`. Architecturally likely safe (Z-Image-Turbo's MMDiT/RoPE lineage is inherently resolution-flexible, per DyPE/FiTv2 and this project's own architecture docs; real precedent exists of practitioners training one LoRA task at two resolutions), but not confirmed as a deliberate choice anywhere in this repo — both yaml files now cross-reference the full research so this stops being a silent mismatch |
| 26 | Multi-turn dialog amnesia in `flows.py` and `planner_backend.py`, dropped structured constraint deltas, disconnected Critic feedback loop, ungrounded sketch prompt conditioning | **Fixed** — see `26_agentic_pipeline_and_multi_turn_sync.md`; conversation history passed to Planner, JSON constraint updates merged into `state.constraints`, Critic suggestions formatted into prompt, sketch prompt grounded in active constraints, Canvas2D UI feedback enabled |
| 27 | Closing PRD open research risks: `_export_zimage` missing `source_caption` schema parity, unconditional `stage=SKETCHING` advance in `flows.py`, "Qwen3.5 needs git-main transformers" resolved via `transformers>=5.2.0` native release, `QwenImageEditPlusPipeline` `strength` kwarg signature inspection, DPO beta/rank sweep recommendations | **Fixed & Researched** — see `27_prd_open_risks_research.md`; schema unified, conditional sketch stage guard applied, dependency pin updated, runtime signature guard in place, full test suite passing (465 passed, 1 skipped) |
| 28 | Production Docker containerization, multi-tier virtualenv isolation (`/opt/venv-inference` vs. `/opt/venv-critic`), fail-fast hardware & CUDA preflight diagnostic engine (`krisna_inference.common.hardware` and `check_hardware.py`), service fail-fast startup guard, `docker-compose.yml` GPU stack, and upgraded installers | **Fixed & Tested** — see `28_docker_isolation_and_hardware_preflight.md`; multi-venv container built, GPU fail-fast diagnostics in service and entrypoint, 16 new unit tests, full test suite passing (481 passed, 1 skipped) |

## Priority ordering, if working through this list top-down
1. **Phase 1's RICO join/count verification** — determines whether the paper's headline data-scale claim is accurate. Still the top open item; nothing fixed in this pass touches it.
2. **Confirm the sketch-tier hyperparameter fixes actually run** — they're syntax-checked and logically sound, but genuinely unverified against `torch` in this environment. Worth a real run before trusting them fully.
3. **Measure Polish-Default's real VRAM footprint on actual hardware** (Phase 12/13 finding) and confirm `model_registry.py`'s corrected `14.0`GB (`REGISTRY`) / `12.0GB+2.0GB RAM` (`LOW_VRAM_REGISTRY`) figures — these are now arithmetic-grounded, not wrong, but still not measured.
4. Everything else remaining is either a real-but-lower-severity fix (dead code paths, missing citations) or a citation/verification task that doesn't change what the system actually does.

---
This closes the six-phase review plan from `00_REVIEW_PLAN.md`. All seven docs (`00`–`06` plus this consolidated `07`) are in `docs/review/`. Phases 12–28 are later follow-up sessions' changelogs against this same set — see those files for anything after this line.
