# Krisna Design-Sync Review — Master Plan

This is an independent audit of the `krisna` monorepo against its own
stated design (`README.md`, `docs/architecture/*`). Goal: verify that
data → training → inference are in sync for every model in the pipeline,
that hyperparameters/optimizations are sound and justified, that the
inference strategy for each model is competitive with current SOTA, and
that every claim is backed by a citation usable in a research paper.

Findings are written **one phase at a time** into `docs/review/`, each as
its own file, so each can be checked and finalized before the next
begins.

## Models / phases in this system

| # | Phase | Model | Trained? | Status |
|---|---|---|---|---|
| 1 | Sketch tier | from-scratch MaskGIT-style bidirectional transformer + VQGAN tokenizer | Yes (Stage1 256px → Stage2 512px) | **Done — see `01_sketch_tier.md`** |
| 2 | Polish tier | Z-Image-Turbo, LoRA + Diffusion-DPO (2-stage: general Pick-a-Pic/HPDv2, then UI-domain) | Yes (LoRA + DPO only, base frozen) | Pending |
| 3 | Planner | Qwen3.5-9B | No — frozen, RAG-only | Pending |
| 4 | Critic | Gemma-4 | No — frozen, on-demand QLoRA adapter only | Pending |
| 5 | Data pipeline (data-forge) | N/A | N/A | Pending — cross-cutting, touched inside each phase above but also needs its own pass (licensing, dedup, ui_first_ratio, sync contract) |
| 6 | Inference orchestrator | SwapOrchestrator, VRAM budget, backends, verifiers | N/A | Pending |
| 7 | Consolidated citations doc | — | — | Pending — merges this repo's existing `RESEARCH_AND_CITATIONS.md` with new findings from phases 1–6 |

## Method for each phase

1. Read the phase's config(s), dataset prep code, model code, and training loop.
2. Cross-check against the corresponding inference backend — same hyperparameters, same tokenizer/codebook, same conditioning shapes, same checkpoint contract.
3. Compare the chosen architecture/training recipe against current SOTA (web search, primary sources only) — is it still a reasonable choice, and does the inference-time sampling strategy match what the literature shows is fast/best for this model family?
4. Flag anything actually out of sync (not stylistic nitpicks) as a **Finding**, severity-tagged.
5. List every source cited, in a format ready to paste into a paper's bibliography.

## Output location

`docs/review/01_sketch_tier.md`, `02_polish_tier.md`, `03_planner.md`,
`04_critic.md`, `05_data_pipeline.md`, `06_inference_orchestrator.md`,
`07_consolidated_citations.md` — each self-contained, cross-linked.

---
Proceeding now with **Phase 1 (Sketch tier)** — see `01_sketch_tier.md`.
Reply to continue to Phase 2 (Polish tier / Z-Image-Turbo DPO), or redirect first.
