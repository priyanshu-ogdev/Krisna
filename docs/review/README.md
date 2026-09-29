# Design-Sync Review — Index

An independent, phase-by-phase audit of this monorepo against its own
stated design: is data → training → inference actually in sync for
every model, are hyperparameters sound and precedent-matched, is the
inference strategy SOTA-competitive, and is every claim backed by a
real, checkable citation. Read `00_REVIEW_PLAN.md` first for method;
everything after is one self-contained finding-set per phase, in the
order they were produced. Later docs correct earlier ones where a
deeper pass changed the conclusion (see `06`'s correction section) —
those are left visible, not silently edited away.

| Doc | Covers |
|---|---|
| `00_REVIEW_PLAN.md` | Method, scope, phase list |
| `01_sketch_tier.md` | MaskGIT-style Sketch tier: data↔train↔inference sync, hyperparameters vs. Chang et al. 2022, real data quantity/quality (RICO/CLAY/Enrico/WebUI) |
| `02_polish_tier.md` | Z-Image LoRA + Diffusion-DPO: data↔train sync, DPO loss/beta verification, diffusion convergence hyperparameters (logit-normal timestep sampling) |
| `03_planner.md` | Qwen3.5-9B (frozen, RAG-only): why frozen, RAG corpus sync, retrieval strategy |
| `04_critic.md` | Gemma-4 31B (frozen, zero-shot): corrects the original plan's "on-demand adapter" assumption, cross-checks the shared critique schema |
| `05_data_pipeline.md` | Cross-cutting data-forge review: closes the Phase 1 RICO-licensing open item, safety/PII stage ordering, domain-balance caveats |
| `06_inference_orchestrator.md` | SwapOrchestrator/VRAM budget — **includes a self-correction**: an initial "critical bug" finding was wrong and is retracted in place, with the real (lower-severity) issue identified instead |
| `07_consolidated_citations.md` | Every citation from Phases 1–6, cross-checked against the repo's existing `RESEARCH_AND_CITATIONS.md`; full open-action rollup with priority order |
| `08_hyperparameter_validation.md` | Validates the two sketch-tier hyperparameter fixes (label smoothing, AdamW betas) against synthetic data shaped like the real config, since a live training run isn't available in this environment — states plainly what this does and doesn't prove |
| `09_synthetic_data_audit.md` | Every dataset traced as real vs. synthetic; identifies the one real synthetic-data touchpoint (VLM-assisted caption densification) and its generalization risk |
| `10_synthetic_data_generalization_fix.md` | Researches and fixes that touchpoint: verifies its purpose against real data (Screen2Words' 6.57-word average captions), closes two real sync gaps (source caption discarded before training; no CFG dropout anywhere), implements + tests both fixes, and addresses the separate general-domain-DPO → UI-domain generalization question |
| `11_scripts_review.md` | Reviews every script under `scripts/` for whether it actually runs the pipeline as documented — finds and fixes a genuinely broken path bug (`verify_schemas.py` never worked), two misleading/stale operator-facing hints for deprecated tiers, a broken doc link, and two stale READMEs; corrects an overstated claim from `02_polish_tier.md` about LoRA composition after tracing the real inference factory code |
| `12_post_upgrade_resync_audit.md` | Follow-up session changelog: closes out the `maskgit_vq` dead-stage fix (upgrades its severity — it was silently emptying the Sketch tier's exported training data on every run, not just dead weight), a `vram_budget.py` doc-accuracy bug, verifies the new `download_weights.py`/`inference-frontend/` artifacts against real training save paths and the real wire contract, and re-checks Phase 2/3/4's model citations against sources published since those phases (closes Phase 4's open Gemma-4 citation gap; upgrades Phase 2's Z-Image-Turbo citation to a primary source; raises one new open Medium-severity finding on Polish-Default's declared VRAM) |
| `13_ram_offload_and_precision_audit.md` | Closes Phase 12's open Polish-Default VRAM finding: traces the actual root cause (a false `quantization="NF4/NVFP4"` claim — the backend has always loaded bf16, deliberately, to match its LoRA's bf16-trained precision), corrects the registry's `vram_gb` from an arithmetic-grounded recomputation, and adds the RAM-offload capability the wrong number had excluded this tier from. Also phase-by-phase re-checks every other backend (Planner/Sketch/Polish-Quality/Critic) for the same declared-vs-actual quantization drift — none found — and confirms the orchestrator/pipeline contract is unaffected by the fix |

## What's fixed vs. still open

Full list with status is in `07_consolidated_citations.md`'s rollup
table (updated through Phase 12). Short version: five items fixed and
verified in the original six-phase pass (label smoothing, AdamW betas, a
stale docstring, a pinned LoRA rank, a VRAM safety-margin capability),
one retracted finding (the "VRAM bug" turned out to be a review error),
two doc-hygiene bugs fixed in `docs/architecture/RESEARCH_AND_CITATIONS.md`
along the way, two more real fixes from `10_synthetic_data_generalization_fix.md`
(caption-style mixing, CFG conditioning dropout), and — as of
`12_post_upgrade_resync_audit.md` — four more real fixes (the
`maskgit_vq` data-loss bug, the VRAM-margin doc bug, the installer's
fabricated artifact paths, the frontend's wrong field names) plus one
new open finding (Polish-Default's declared VRAM may be under-sized —
needs a real hardware measurement) and two citation gaps closed
(Gemma 4, Z-Image-Turbo upgraded to primary source). **As of
`13_ram_offload_and_precision_audit.md`, that Polish-Default finding is
closed**: root-caused to a false quantization claim, `vram_gb` corrected
to an arithmetic-grounded 14.0 (full) / 12.0+2.0GB-RAM (low-VRAM,
offload-enabled) estimate, RAM offload added and wired through, and all
four other backends re-checked for the same drift class (none found).
**Top remaining items:** (1) verifying the exact real usable-image count
for the Sketch tier once the RICO-license/dedup/join interaction is
exercised against a real pipeline run, (2) measuring Polish-Default's
(and the other tiers') actual VRAM footprint on real hardware to convert
these grounded estimates into confirmed numbers.
