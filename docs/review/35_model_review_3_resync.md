# 35 — Model 3/5 resync: comparing Polish-Default against Models 1 and 2

A second, comparative pass over Polish-Default specifically asked to
check consistency against the Planner and Sketch reviews. Systematic
check: "does `run()`'s own parameter list have a dead one, like
`planner_output` did for Sketch?" — answer: no, `polish_default_backend.py`'s
`run()` has no dead parameters. But broadening the same question to
"does the SAME Planner-reasoning-note enrichment pattern exist
elsewhere for this tier, and if not, should it?" found a real, genuine
gap — and researching the right fix properly, rather than copy-pasting
Sketch's solution, surfaced an important architectural difference that
would have made a blind copy-paste wrong.

## The gap: `_synthesize_dpo_prompt` had the same missing enrichment as Sketch, for a different reason

`flows.py::_synthesize_dpo_prompt` — the function that builds the
prompt for BOTH Polish-tier generation (`finalize()`) and DPO
preference-pair training data (`critique_pass()`) — builds its prompt
from `state.conversation_history` and `state.constraints` only. It
never incorporated the Planner's `reasoning_note`, the same gap already
found and fixed for `sketch_backend.py`'s `run()` in `docs/review/32`.

**The architectural difference that mattered**: Sketch's `run()`
receives `planner_output` directly, live, from the SAME conversational
turn — a simple parameter read. `_synthesize_dpo_prompt` only has
access to persisted `DesignState`, which had no field to carry a
reasoning_note forward from whenever the Planner last produced one to
whenever `finalize()`/`critique_pass()` is later called (these aren't
called every turn — finalize happens once the user is ready). Fixing
this required an actual schema addition, not just a read.

Added `DesignState.last_planner_reasoning_note: str | None`, placed in
the same category as `revision`/`created_at`/`updated_at` — explicitly
NOT part of PRD §5.1's literal wire contract (popped out in `to_wire()`
the same way those three already are), since `DesignState`'s own
docstring states it's "field-for-field" with the PRD and this field has
no PRD-defined counterpart. `conversational_turn()` now persists it
from each turn's `design_state_delta.reasoning_note`, but only when
non-empty — a turn where the Planner has nothing new to add must not
erase a meaningful note from an earlier turn.

## Why the bound had to be researched, not reused

Sketch's fix capped the reasoning_note at 160 characters specifically
because CLIP's tokenizer truncates silently at 77 tokens. Rather than
reusing that number here, checked what Z-Image-Turbo's actual text
encoder is: **Qwen3-4B**, confirmed directly from its real
`text_encoder/config.json`, with `diffusers`/`DiffSynth` enforcing a
practical `max_sequence_length=512` tokens — roughly 6-7x CLIP's
~77-token budget. Reusing Sketch's tight cap here would have been
unnecessarily conservative, under-using real available context for no
reason. Used a generous 300-character defensive ceiling instead (guards
only against a Planner that ignored its own "≤2 sentences" system-prompt
instruction, not against a truncation risk that doesn't meaningfully
exist at this tier's real context budget).

## Why this also affects DPO training data, deliberately

`_synthesize_dpo_prompt`'s own existing docstring states it's shared
between `finalize()` and `critique_pass()` specifically "so the LoRA
trains on the distribution it will actually see at inference time."
This enrichment therefore also reaches `PreferencePair.prompt` for DPO
training going forward — a deliberate, consistent extension of that
existing design intent (richer prompts in both contexts, not a
divergence between them), not an accidental side effect.

## What else was checked in this comparative pass, and found already correct

- `polish_default_backend.py::run()`'s full parameter list
  (`handoff_image_ref`, `constraints`, `prompt`, `seed`) — confirmed
  every one is actually read in the method body, unlike Sketch's
  now-fixed `planner_output`. No dead parameter here.
- Re-confirmed `docs/review/34`'s caption-mixing and LoRA alpha/rank
  fixes are still intact and the full suite is green after this pass's
  additional changes.

## Test coverage

5 new tests in `test_flows_prompt_synthesis.py::TestReasoningNoteEnrichment`
(enrichment present/absent, the generous-bound proof, the defensive
cap, and the `to_wire()` exclusion). 2 new/updated tests in
`test_flows.py` — one extending an existing integration test with a
direct assertion on the already-present-but-previously-unasserted
`reasoning_note` input, one new test proving the "don't erase on an
empty turn" behavior through the real `conversational_turn()` entrypoint,
not just the pure function in isolation. Full repo suite: 553 passed, 1
skipped, 0 failures.
