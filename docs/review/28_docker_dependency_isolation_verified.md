# 28 — Docker/dependency-isolation research, closed out

Continues `27_prd_open_risks_research.md`, which flagged three items as
"recommended, not yet executed" or "not verified, no network access."
This pass had real network access (pypi.org, github.com, files.pythonhosted.org)
and closed all three with direct, sourced verification rather than more
speculation.

## 1. Stale "Planner needs git-main transformers" docstrings — FIXED

Doc 27 already updated `requirements-inference.txt` and
`docs/architecture/RESEARCH_AND_CITATIONS.md` to reflect
`transformers>=5.2.0` (Qwen3.5's native, tagged support), but four LIVE
source locations still had the pre-fix claim, causing real drift between
what the requirements file said and what the code/docs claimed:

- `inference/src/krisna_inference/backends/critic_worker.py` (module docstring)
- `inference/src/krisna_inference/backends/critic_backend.py` (module docstring)
- `inference/src/krisna_inference/backends/planner_backend.py` (module docstring)
- `inference/README.md` ("Critic tier isolation" section)

All four corrected this pass. `docs/review/04_critic.md` — an earlier,
numbered historical review entry — was deliberately left unedited,
matching this project's own convention (see PRD's "superseded, not
merged" framing for historical docs) of preserving what was believed at
the time rather than retroactively rewriting review history; doc 27 and
this doc supersede it.

## 2. Why the Critic subprocess/venv split is STILL necessary — re-derived with the real reason

The split's original justification (an unresolvable git-main-vs-5.5.0
transformers conflict) no longer holds, now that Qwen3.5 has tagged PyPI
support. Re-investigated from scratch rather than assuming the split is
now unnecessary:

- Cloned `huggingface/diffusers` directly (shallow, `@main`) and read
  its actual `setup.py`: declared floor is `transformers>=4.41.2` — no
  upper bound, no conflict with `transformers>=5.2.0` or
  `transformers==5.5.0`.
- Found the REAL reason `transformers==5.5.0` is pinned exactly, via
  `unslothai/unsloth-zoo` PR #1227's own measurement: Unsloth's
  prequantized `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` checkpoint loads
  with all 352/352 modules correctly dequantized (`quant_state` present)
  on `transformers==5.5.0`, but **0/352** on `transformers==5.17.0` — a
  real, forward-pass-breaking regression, tracked upstream as
  `unslothai/unsloth#9867`, `#10010`, `#10017`, `#10276`.
- This means the exact pin is deliberate regression-avoidance, not an
  arbitrary compatible version. **The split now earns its keep for a
  different, better reason than originally stated**: it decouples this
  regression-avoidance pin from whatever transformers/diffusers versions
  the other three tiers evolve onto over time. Collapsing it into one
  venv would either cap every tier at 5.5.0 indefinitely, or risk a
  future version bump (made for Planner/Sketch/Polish's benefit)
  silently re-breaking Critic.
- Updated all four docstrings/README section above to state this
  correctly, replacing the resolved conflict-based reasoning.

## 3. `unsloth`/`unsloth_zoo` version floor — verified and pinned

`requirements-critic.txt` previously left `unsloth`/`unsloth_zoo`
completely unpinned. Checked whether this was actually safe:

- `unslothai/unsloth` issue #4022 documents that an EARLIER unsloth_zoo
  release capped `transformers<=4.57.6` — strictly below 5.0.0, which
  would make `transformers==5.5.0` unresolvable entirely.
- `unslothai/unsloth` PR #11210 (raising unsloth's own installer floor)
  confirms, as of the real, dated, current release **unsloth_zoo
  2026.9.5**, the declared ceiling is `transformers<=5.5.0` — already
  relaxed past the old `<=4.57.6` blocker, and already permits our exact
  pin (`==5.5.0`) without needing the further relaxation proposed in PR
  #1227 (which was still unmerged/unreleased as of the evidence found in
  this pass — not depended on).
- Pinned `unsloth_zoo>=2026.9.5` and `unsloth>=2026.9.7` (the paired
  current stable release per the same PR thread) in
  `requirements-critic.txt`, rather than leaving both floating and
  hoping pip lands on a compatible pair.

## 4. `diffusers @ git+main` — pinned to a verified commit

Floating `@main` branch pins are a real reproducibility risk (a
container rebuilt tomorrow could silently resolve a different upstream
commit, with no diff visible anywhere in this repo). Resolved by:

- Shallow-cloning `diffusers` and confirming the specific commit
  (`cc8644b447d8f11074d3df06d0ee0e3e7c91bf75`) contains BOTH pipeline
  classes this project depends on
  (`src/diffusers/pipelines/z_image/pipeline_z_image_img2img.py`,
  `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py`),
  each with real upstream test coverage
  (`tests/pipelines/z_image/test_z_image_img2img.py`,
  `tests/pipelines/qwenimage/test_qwenimage_edit_plus.py`).
- Directly read `ZImageImg2ImgPipeline.__call__`'s real signature at
  this commit and confirmed `strength: float = 0.6` is its actual
  default — matching `polish_default_backend.py`'s own default exactly
  (that default was chosen from an official usage example in this
  project's prior review pass, not yet cross-checked against the literal
  source; now it has been).
- Verified the pin installs cleanly via `pip download --no-deps` against
  this exact commit SHA.
- Updated `requirements-inference.txt` to this pinned commit. Bumping it
  going forward should be a deliberate, verified action, not a re-float
  to `@main`.

## Remaining open item (genuinely unresolved, not guessed at)

Whether `diffusers`'s pinned commit has any transitive requirement on a
`transformers` version newer than `5.5.0` for the SPECIFIC code paths
Sketch/Polish-Default/Polish-Quality actually exercise (as opposed to its
declared, very loose `>=4.41.2` floor) is not fully verifiable without a
real GPU + full install + exercising those code paths — genuinely out of
reach in this sandbox. Flagged rather than assumed fine; if a future
real-hardware run hits an import/attribute error tracing to a
transformers version mismatch in the main venv, this is the first place
to look.
