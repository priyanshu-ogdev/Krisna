# Phase 4 — Critic (Gemma 4 31B Dense, fully frozen, zero-shot VLM-as-judge)

## Correction to the Phase-0 plan
The master plan labeled this "frozen base + on-demand QLoRA adapter."
That's wrong, verified against the actual `models/` directory: there is
no Critic adapter artifact anywhere in this repo (only
`planner_rag_index/`, `dpo_checkpoints/`, `polish_default_lora/`,
`sketch_tier/` exist). The Critic is **fully frozen, zero-shot-only** —
no adapter, on-demand or otherwise, in the active pipeline.

## What exists and what doesn't
- **Active, live:** `critic_worker.py` (isolated-subprocess model runner) + `critic_backend.py` (orchestrator-facing wrapper) + `critique_adapter.py` (output normalizer). Loads `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`, pre-quantized NF4, zero-shot VLM judging — no training, no fine-tune, no adapter load path in this code at all.
- **Deprecated, correctly labeled (not a sync bug):** `training/src/krisna_training/critic/` implements real, working QLoRA-via-Unsloth training for the same base model. It is explicitly out of the active pipeline — emits `DeprecationWarning` on import, called out as deprecated in `training/README.md`'s own component table, kept only as reference code in case a future PRD revision un-freezes this tier. This is the **correct way** to leave unused-but-functional code in a repo, and stands in useful contrast to Phase 1/2's silent orphans (`maskgit_vq` encoding, `s08_5_dpo_encoding.py` output) — those should be fixed to look like this: either wired in or explicitly, loudly marked dead.

## Cross-cutting schema consistency — verified in sync
`data-forge/configs/schemas/critique_output.json` (four dimensions: `visual_hierarchy`, `readability`, `layout_consistency`, `brand_alignment`, each `_score`+`_note`, plus `overall_score` and `suggested_edits`) is the **same shape** used in two independent places:
1. UICrit's human-labeled critiques, normalized into this shape for the Planner's RAG corpus (`uicrit_ingest.py::to_critique_output_dict`, verified in Phase 3).
2. The Critic's own live, model-generated judgments, normalized into the same shape via `critique_adapter.py::from_gemma_output()` → `CritiqueResult` (pydantic-validated, not trusted raw from the subprocess).

Both paths converge on one shared `CritiqueResult` schema, distinguished only by `critique_source` (`"uicrit_human"` vs `"gemma4_31b_frozen"`), so anything downstream (DPO ranking, preference-pair store) genuinely doesn't need to know which produced a given critique. This is real, verified design consistency across two phases reviewed independently — not just an assertion.

## Process isolation — verified, not just claimed
`critic_backend.py` and `critic_worker.py` run the Critic in a fully separate subprocess/venv (`./venv-critic`) specifically because `transformers==5.5.0` (Unsloth's Gemma-4 pin) conflicts with the Planner's requirement of `transformers` built from git main for Qwen3.5 support. This is a real, unresolvable-in-one-venv dependency conflict, not an overcautious design choice — confirmed by checking `training/critic/__init__.py`'s docstring (isolation reasoning explicitly stated as unaffected by the freeze, since it's about version pins, not training) and consistent with what Phase 3 found about the Planner's own dependency requirements.

## No hyperparameters to review
Since the Critic ships frozen with no active training or adapter loading, there is no training hyperparameter surface for this phase — the deprecated `critic_qlora_train.yaml` (`lora_r=16`, `lora_alpha=16`, `lr=1e-4`, batch 1 × grad-accum 8, 2000 steps) is preserved reference config only, not something currently trained against. Not reviewing it against SOTA QLoRA precedent since it isn't live — flagging this explicitly rather than silently skipping, so it's clear the omission is deliberate.

## Findings summary

| Severity | Finding |
|---|---|
| Info | Phase-0 plan's "on-demand QLoRA adapter" framing was inaccurate — corrected here; Critic is fully frozen/zero-shot |
| Info | Deprecated Critic training code is a model example of how to retire unused code (loud deprecation warning + doc table entry) — contrast against Phase 1/2's silent orphans |
| Info | Critique schema is genuinely, verifiably shared and in sync across UICrit-sourced (Planner RAG) and Gemma-4-sourced (Critic) critiques |
| Info | Process isolation (`venv-critic`) is a real, necessary fix for a genuine dependency conflict, not overengineering |
| N/A | No hyperparameters to review — nothing trains in this phase |

## Citations (Phase 4)
1. Gemma Team, Google DeepMind — Gemma model family (cite the specific Gemma-4 technical report/model card directly in the paper; not independently verified against a primary source in this pass — flag as an action item if Gemma-4-specific citation detail is needed).
2. Unsloth — VLM LoRA fine-tuning API (`FastModel.get_peft_model`, `finetune_vision_layers`/`finetune_language_layers`/etc.), docs.unsloth.ai/basics/vision-fine-tuning — cited in-repo for the deprecated training path; real and verifiable, relevant only if that tier is ever revived.
3. UICrit — Google Research, `google-research-datasets/uicrit` (already cited in Phase 3; relevant here again as the schema this phase's live output is normalized to match).

---
Next: Phase 5 — Data pipeline cross-cutting review (licensing/dedup/ui_first_ratio), or Phase 6 — Inference orchestrator (SwapOrchestrator, VRAM budget). Which would you like first?
