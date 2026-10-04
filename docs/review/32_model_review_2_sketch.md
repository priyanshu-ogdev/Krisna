# 32 — Model 2/5 deep review: Sketch tier (from-scratch MaskGIT transformer)

Systematic review of the Sketch tier: every training↔inference
connection point traced, the Planner→Sketch handoff specifically
audited (per this session's explicit ask to "connect with model 1
thoroughly"), and real bugs found and fixed — one significant
(silently-discarded Planner context), one a test-infrastructure
landmine that would have caused a very confusing false "hang" report
on a future, unrelated test.

## Real bug #1: `planner_output` accepted, never used — the same class of bug already found once

`SketchBackend.run(self, planner_output: dict | None = None, ...)`
accepted this parameter on every single conversational turn (passed by
`swap_orchestrator.run_conversational_turn`), but nothing in the method
body ever read it. This is the exact same "accepted but silently
ignored" pattern already found and fixed once this project
(`polish_default_backend.py`'s `handoff_image_ref`) — confirmed by
`grep`, the name appeared exactly once, in the signature.

**What was actually being thrown away**: `planner_output`'s
`design_state_delta.reasoning_note` — the Planner's own ≤2-sentence
synthesis of design intent (e.g. "User wants a warmer, friendlier feel
with rounded corners"). This is richer, design-vocabulary-appropriate
prose that the terse `style`/`palette`/`layout_hints` constraint tags
alone can't capture — a genuine Model-1-to-Model-2 connection this tier
was silently discarding, not a cosmetic gap.

**Fix, with a real risk considered and bounded**: folded `reasoning_note`
into the CLIP-conditioning prompt text, but capped to 160 characters.
Checked `verifiers/common.py::embed_text` first — CLIP's tokenizer
truncates SILENTLY at 77 tokens (`truncation=True`, no error, no
warning). Appending an unbounded reasoning_note after the user's own
message could, on a long turn, push the user's actual message itself
past the truncation point — a worse outcome than not using
reasoning_note at all. The cap keeps the addition strictly additive: it
enriches conditioning without risking the primary signal (the user's own
words) being silently dropped.

Guards against malformed input (`planner_output` not a dict, or a dict
missing the expected nested shape) degrade gracefully rather than
crashing generation mid-turn — matching this codebase's established
"flag rather than guess" posture for a JSON-shaped value produced by a
frozen model whose output isn't grammar-constrained (see
`planner_backend.py`'s own generate-validate-retry loop for the same
posture one layer up).

## Real bug #2: a test-fixture landmine that manifests as a hang, not a clean failure

While adding a regression test that calls `SketchBackend.run()` twice in
one test (needed to exercise the fix above across multiple inputs), the
test suite **hung** rather than failing. Root cause, traced fully: the
shared test fixture `_make_backend_with_fakes` set
`fake_module.parameters.return_value = iter([fake_param])` — a
**one-shot iterator**. Every existing test in this file called
`backend.run()` exactly once, so this was never exercised twice and the
bug stayed dormant. On a second call, `next(self._model.module.parameters())`
hits an exhausted iterator and raises `StopIteration` — which, raised
inside a thread run via `asyncio.to_thread`, triggers a known
asyncio/PEP 479 interaction (`StopIteration` cannot be raised into a
`Future`) that manifests as a **hang**, not a clean test failure. This
is a nasty failure mode to debug blind — it looks exactly like the
*code under test* is hanging, when the actual bug is in the test
fixture's mock setup. Fixed with `side_effect=lambda: iter([fake_param])`
so every call gets a fresh iterator. Documented plainly in the fixture
itself so this doesn't get rediscovered the hard way by whoever writes
the next multi-call test in this file.

## Training↔inference connection verification — traced, no drift

- **Checkpoint format**: `training/sketch/checkpoint_io.py::save_checkpoint`
  and `maskgit_model.py::from_checkpoint` — verified the exact key set
  (`model_state_dict`, `config`, `grid_h`, `grid_w`, `mask_token_id`)
  matches on both sides, with an explicit "do not rename" comment on the
  training side. No drift.
- **Architecture definition lives in exactly one place**: inference
  rebuilds the model via `build_model(ckpt["config"])` — the same
  function training uses — rather than duplicating the architecture
  definition on the inference side. This means the two sides cannot
  silently diverge on model structure the way they theoretically could
  if inference had its own copy of the transformer definition.
- **Prompt embedding space**: `sketch_backend.py` loads the SAME CLIP
  embedder singleton (`verifiers.common.get_clip_embedder`) that
  training's `make_collate_fn` uses — guaranteeing the embedding space
  matches by construction, not by convention. Verified `embed_text`
  returns a proper `[1, dim]` batch-shaped tensor on both call sites
  (confirmed earlier in this project's review), and `prompt_dim`
  fallback (768, CLIP ViT-L/14's real dimension) matches what every
  real training config actually uses.
- **`num_rounds`/`guidance_scale`**: purely inference-time sampling
  hyperparameters (MaskGIT's progressive-unmasking schedule), not
  coupled to any specific training-time value — confirmed this is
  correctly the case, not something silently mismatched.

## Cross-cutting note for later models

Every review so far in this series (Planner, now Sketch) has turned up
at least one "parameter accepted, never used" bug at a boundary between
two models/tiers — the exact seam where information is supposed to flow
from one stage to the next. Recommend checking this specifically for
Model 3 and Model 4 (both Polish tiers) and Model 5 (Critic): confirm
every parameter a `run()` method accepts is actually read somewhere in
its body, not assumed to be from the signature alone.

## Test coverage

11 tests in `tests/inference/test_sketch_backend_prompt.py` (up from 5):
the fix's core behavior (reasoning_note folded in, capped, malformed
input handled), plus the shared fixture fix. Full inference suite (211
tests) passes, no hangs.
