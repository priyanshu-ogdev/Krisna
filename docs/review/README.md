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
| `14_safety_gate_wiring.md`, `15_data_forge_finalization.md` | Later sessions (see repo's own commit history/conversation log): `14` wires `VerifierStack.safety_gate()` into the finalize flow after finding it was defined but never called anywhere — an unsafe image could previously finalize successfully with no enforcement. `15` verifies every model's training data comes from real, licensed(-or-flagged) sources (dataset registry found to be the most thoroughly-audited part of the whole project — no gaps found), re-traces the synthetic-caption pipeline end-to-end and confirms it's genuinely wired (all five links from VLM recaption through CFG dropout live), and closes out a second orphaned-producer stage (`s08_5_dpo_encoding.py`) that Phase 2 had already flagged but that was still running by default |
| `16_preprocessing_ordering_audit.md` | Found and fixed the highest-severity issue in this entire review: `s04_5_escalation` ran *after* `s05_recaption`/`s06_structure` in the real execution order, so every record the two-tier safety escalation system successfully rescued (borderline → confirmed safe) was silently, permanently dropped from the training pool — invisible to every prior phase's pairwise stage-filter checks, only found by reading the actual multi-phase control flow end to end. Also gives a direct opinion on the three previously-"still open" items: recommends deleting (not wiring) `s08_5_dpo_encoding.py`; confirms the RICO join count and VRAM measurements remain genuinely blocked on hardware/network access unavailable in this environment |

## What's fixed vs. still open

Full list with status is in `07_consolidated_citations.md`'s rollup
table (updated through Phase 16). Short version: five items fixed and
verified in the original six-phase pass (label smoothing, AdamW betas, a
stale docstring, a pinned LoRA rank, a VRAM safety-margin capability),
one retracted finding (the "VRAM bug" turned out to be a review error),
two doc-hygiene bugs fixed in `docs/architecture/RESEARCH_AND_CITATIONS.md`
along the way, two more real fixes from `10_synthetic_data_generalization_fix.md`
(caption-style mixing, CFG conditioning dropout), four more real fixes
as of `12_post_upgrade_resync_audit.md` (the `maskgit_vq` data-loss bug,
the VRAM-margin doc bug, the installer's fabricated artifact paths, the
frontend's wrong field names), the Polish-Default VRAM/quantization fix
in `13`, a safety-gate wiring fix in `14`, a second orphaned data-forge
stage disabled in `15`, and — as of `16_preprocessing_ordering_audit.md`
— the single highest-severity fix found in this whole review: an
execution-order bug that silently dropped every escalation-rescued
record from the training pool, invisible to every earlier phase's
pairwise stage-filter checks. **Top remaining items:** (1) verifying
the exact real usable-image count for the Sketch tier once the
RICO-license/dedup/join interaction is exercised against a real pipeline
run, (2) measuring Polish-Default's (and the other tiers') actual VRAM
footprint on real hardware, (3) `s08_5_dpo_encoding.py`'s wire-vs-delete
decision — `16` recommends deletion but hasn't executed it, to avoid a
rushed multi-file change under this review's own time constraints.
