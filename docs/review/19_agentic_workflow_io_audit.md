# Phase 19 — Agentic Workflow I/O Audit (Planner→Sketch→Polish→Critic)

Line-by-line trace of the actual data contract at every hop in the
conversational pipeline — not each function's own correctness in
isolation, but whether what one stage produces is what the next stage
actually needs. Two of the three findings here are real, always-firing
bugs in the live service path, not edge cases.

## Finding 1 (highest severity): Finalize crashed on every real call

`flows.finalize()`'s `handoff_hook` parameter — meant to decode Sketch's
VQ tokens into a real pixel image before handing off to Polish — defaults
to identity passthrough when not supplied. `sketch_handoff.py`'s
`make_vq_decode_handoff()` implements this decode correctly and
completely, but was **never imported or called anywhere outside its own
tests**. `service.py`'s real call to `flows.finalize()` passed no
`handoff_hook` at all.

Consequence, traced precisely: `state.sketch_tokens.vq_tokens` is a
`"blob://tokens_xxx.json"` reference (`BlobStore.save_tokens()` writes a
JSON file of token IDs). With the identity-passthrough default, that
same string gets passed straight through as `handoff_image_ref` to the
Polish backend, which calls `store.load_image()` on it →
`PIL.Image.open()` on a JSON text file → **guaranteed
`UnidentifiedImageError`, every single time**, not a hypothetical edge
case. This was the only code path `service.py` used.

**Fix**: wired `make_vq_decode_handoff()` into `service.py`'s real
backend-startup block, loading a `VQTokenizer` from
`KRISNA_VQGAN_CHECKPOINT`/`KRISNA_VQGAN_CONFIG` (new env vars,
explicit rather than reaching into the asynchronously-loaded
`SketchBackend`'s internals) and grid dims from
`KRISNA_SKETCH_GRID_H`/`KRISNA_SKETCH_GRID_W`. If those aren't
configured, `_handoff_hook` now raises a clear, actionable
`RuntimeError` at first use instead of silently passing a token blob
through to crash cryptically inside PIL three layers deeper.

**A second, hidden crash site the same fix resolved**: `flows.py`'s
`_run_verifier_stack()` loads the sketch image for `handoff_consistency`
scoring with `except FileNotFoundError` only — but before this fix,
that same raw token blob would raise `PIL.UnidentifiedImageError`, which
that clause didn't catch. Already moot with the real fix in place, but
broadened to `except Exception` anyway, matching this file's own
established "degrade to `None`, don't abort the whole finalize pass"
philosophy for optional verifier inputs.

## Finding 2: raw JSON glued onto every planner chat message

`PlannerBackend._extract_json_delta()` locates the JSON delta's
character span within the raw model output purely to parse it, then
discards that span. The full raw text — JSON object still attached —
flowed unmodified through `mock_output_text` (also renamed to
`reply_text` in this pass — a real backend returning a key literally
named `mock_output_text` was confusingly named even though it wasn't
functionally wrong, since Mock and real backends deliberately share this
key by design) → `flows.py`'s `state.append_turn("planner", ...)` →
`conversation_history[].content` → the chat UI, which renders it
verbatim. Every planner turn showed the user a reply with a raw JSON
object glued onto the end.

**Fix**: `_extract_json_delta()` now returns the match span too;
`run()` strips it from the text before returning. Verified directly
against the real code (pure string/JSON logic, no `torch` needed) — and
caught a real mistake in my own first attempt at this fix: the method's
final success-path `return parsed, None` sat just outside the first
edit's replacement span and kept returning a 2-tuple after every other
return statement had been updated to 3-tuple, causing an unpacking
`ValueError` the moment a well-formed reply was parsed. Found by running
the actual function against a real example, not by re-reading the diff.

## Finding 3: critique results were fetched, stored, never shown

Clicking "Critique" in the frontend correctly POSTs, correctly receives
`critique.result` (matching `design_state.py`'s `CritiqueResult` schema
exactly — verified), and correctly stores it in `sessionState` — but
`applySession()` never called anything to render it. Zero visible
feedback to the user that a critique happened, let alone what it said.

**Fix**: added `renderCritiquePanel()` + a panel in `index.html`, wired
into `applySession()`, matching `renderFinalizePanel()`'s existing
score-bar styling and the real `CritiqueResult` shape (`overall_score`,
`dimensions{score,note}`, `suggested_edits`).

## What was checked and found already correct

Traced the rest of the Sketch→Polish→Critic chain key-by-key: both
Polish backends consistently return `image_ref`; the subprocess IPC
protocol between `critic_backend.py` and the isolated-venv
`critic_worker.py` matches on both sides (`{"cmd": "run", "params":
{...}}` in, `HANDLERS["run"]` dispatch, `critique_result` out);
`critic_worker.py`'s `_build_prompt()` and
`training/critic/dataset.py`'s `build_critic_prompt()` are **verified
word-for-word identical** (executed both, diffed the output strings
character-by-character, not eyeballed) despite the docstring's own "MUST
stay identical" claim being exactly the kind of assertion this review
doesn't take on faith elsewhere; the prompt correctly instructs Gemma to
use the same four dimension names `critique_adapter.py`'s
`_VERIFIER_TO_DIMENSION` expects, resolving what had been an open
question from the previous session's summary. One false alarm caught
and corrected in my own process: initially flagged `constraints`/`image`
as possibly undefined in `critic_worker.py`'s `_run()` based on a
truncated view of the file — re-checked against the actual function
start and both are correctly unpacked from `params` two lines up.

## Regression note

Both Finding 1 and Finding 2's fixes (along with most of Phase 17 and
18's work) were found completely missing from a later uploaded working
copy partway through this review and had to be fully reapplied — see
the repo's own conversation log for the incident. Restated here because
it's the kind of thing a reader of just this doc wouldn't otherwise
know to check for: **verify these fixes are actually present in
whatever copy of this repo you're looking at**, don't assume this doc
existing means the code does too.
