# Phase 26 — Agentic Pipeline Multi-Turn Synchronization & Training-Inference Alignment

Audit and resolution of data/context handoffs across the conversational agentic pipeline (Planner ↔ Sketch ↔ Polish ↔ Critic ↔ User Interface) and verification of training weight compatibility before training execution.

---

## 1. Context & Motivation

Before launching full-scale model training and end-to-end evaluation, the entire inference and training synchronization layer was subjected to a rigorous audit:
1. Did the training checkpoints produce artifacts that inference backends could load without schema/precision mismatches?
2. Did the Planner maintain conversational memory across multi-turn exchanges, or did it suffer from amnesia?
3. Were structured design updates emitted by the Planner actually saved into the session's `DesignState`?
4. Did reviewer critique from the Critic tier feed back into subsequent generation turns?
5. Did the user interface provide clear visualization of the model's design reasoning and enable 1-click agentic iteration?

---

## 2. Findings & Resolutions

### Finding 1 (Critical): Multi-Turn Dialog Amnesia in `flows.py` & `planner_backend.py`
- **Root Cause**: In `flows.conversational_turn`, `orchestrator.run_conversational_turn` was invoked with only `message=user_message`. Dialog history in `state.conversation_history` was never forwarded. In `PlannerBackend._generate()`, messages only contained `[system_prompt, user_message]`.
- **Impact**: On turn 2+, the Planner had zero memory of previous constraints or refinements discussed with the user.
- **Resolution**: `flows.conversational_turn` now extracts `conversation_history = [t.model_dump() for t in state.conversation_history]` and forwards it. `PlannerBackend` injects up to the last 6 turns into the Qwen chat template as alternating user and assistant turns.

### Finding 2 (High): Dropped Structured Constraint Updates
- **Root Cause**: `PlannerBackend` outputs a JSON delta with `constraint_updates: {"style": "...", "palette": [...], "layout_hints": "...", "locked_regions": [...]}`. `flows.conversational_turn` read only `reply_text` and dropped the delta.
- **Impact**: Constraints discussed and updated by the Planner were never saved into `state.constraints`.
- **Resolution**: Extracted `design_state_delta` from planner output and merged `style`, `palette`, `layout_hints`, and `locked_regions` directly into `state.constraints`.

### Finding 3 (High): Critic Feedback Disconnected from Planner
- **Root Cause**: After running a critique pass, Gemma-4 evaluation scores and suggested edits were written to `state.critique.result`, but were never supplied to the Planner on subsequent design turns.
- **Impact**: The user or agent could request a critique, but the Planner had no knowledge of the reviewer's feedback when formulating the next sketch.
- **Resolution**: `state.critique.result` is now passed as `prior_critique` into `flows.conversational_turn` and forwarded to `PlannerBackend`. `PlannerBackend._build_system_prompt()` formats overall score, dimension notes, and suggested edit instructions directly into the system prompt.

### Finding 4 (Medium): Ungrounded Sketch Prompt Conditioning
- **Root Cause**: `SketchBackend.run` used `prompt_text = message or "UI design"`. When turn 2 was a brief phrase like "make buttons green", the CLIP text embedder received no UI context.
- **Impact**: Sketch generation drifted from established design style and layout.
- **Resolution**: Enriched prompt text using active constraints (`f"{message} ({constraint_summary})"`), grounding the CLIP conditioning vector in active design constraints.

### Finding 5 (Medium): Generic Fallback in Finalize Prompt
- **Root Cause**: If `prompt` was None in `flows.finalize`, polish tiers fell back to generic placeholders.
- **Impact**: Finalize render lost user conversational intent.
- **Resolution**: Synthesized an enriched prompt from conversation history and active constraints when none was explicitly provided, flowing into both Polish backends and the Verifier Stack.

### Finding 6 (Medium): Frontend Missing Agentic Iteration Loop
- **Root Cause**: Suggested edits in the critique panel were unformatted JSON strings with no interactive action.
- **Resolution**: Formatted critique suggested edits with region badges (`[hero_cta]`, `[navbar]`) and added an interactive **"✨ Iterate with Critic Edits"** button and individual `Apply ↳` quick chips that insert recommendations directly into the chat input.

---

## 3. Verification

All 441 unit and integration tests across data-forge, training, and inference passed with zero failures:

```powershell
python -m pytest -q
# 441 passed, 3 warnings in 33.91s
```
