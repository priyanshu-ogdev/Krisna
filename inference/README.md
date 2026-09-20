> **Split from the original krisna-orchestrator README** as part of the
> training/inference package separation — see
> `docs/architecture/DIRECTORY_LAYOUT.md`. This covers the serving/
> orchestration side (`krisna_inference`). For training code (sketch
> tier, Z-Image-Turbo LoRA, the data-forge sync bridge, DPO data prep),
> see `../training/README.md`.

# Krisna Swap Orchestrator

Swap Orchestrator + Design State Manager for **Krisna's Agentic Design
System** — the coordination backbone described in §5.1 (Design State
object), §5.2 (Critique Adapter contract), §5.3 (sequence flows), and
§7.4 (formal swap-orchestration state machine). This README targets the
pyproject.toml-labeled "PRD v10" baseline plus the **final, no-RLHF-loop
PRD revision** on top of it: the Planner, Qwen-Image-Edit-2511, and
Gemma-4 Critic all ship **frozen**; only the Sketch Tier (from scratch)
and Z-Image-Turbo (LoRA + Diffusion-DPO) are actually trained. Sections
below are labeled FROZEN / TRAINED accordingly — if you're looking for
"which tiers does this project fine-tune," that's the fast answer.

## What this is (and isn't)

This is the **orchestration layer**: it decides *which model tier is
resident on the GPU right now*, enforces the 16–24GB VRAM envelope (§3),
handles OOM with retry + fallback-tier + recovery (see
`swap_orchestrator.py`'s module docstring for the exact policy), and holds
per-session Design State (§5.1) with optimistic-concurrency persistence.

It does **not** load real model weights *in the core orchestrator package*.
`model_registry.py` defines a `ModelBackend` interface with two reference
implementations plus a real one:
- `MockBackend` — used by every orchestrator/state-machine test and by the
  service by default. Supports scripted latency and OOM injection so the
  orchestrator's failure paths are actually exercised, not just the happy
  path.
- `TorchBackend` — a deliberately-`NotImplementedError` placeholder,
  kept as the minimal interface reference.
- `inference/factory.real_backend_factory` — the actual working
  implementations (Qwen3.5 planner **[FROZEN — RAG]**, Z-Image-Turbo
  **[TRAINED]**, Qwen-Image-Edit-2511 **[FROZEN]**, Gemma 4 critic
  **[FROZEN]**, checkpoint-pluggable sketch tier **[TRAINED, from
  scratch]**). See "Inference layer" below. Swap it in via
  `SwapOrchestrator(backend_factory=...)` or `KRISNA_USE_REAL_BACKENDS=1`
  — nothing in the orchestration layer itself changes (the orchestrator
  is training-status-agnostic by design: it only ever calls
  `load()`/`run()`/`unload()` through the common `ModelBackend`
  interface).

### Canonical Specification (§7.4)

The canonical state machine specification is formalized in [docs/PRD.md §7.4](file:///d:/Krisna/docs/PRD.md).
The transition table implemented in `state_machine.py` matches that canonical
specification exactly: `IDLE_RESIDENT` → `SWAPPING_TO_POLISH` → `POLISH_RESIDENT`
→ `SWAPPING_BACK_FROM_POLISH` → `IDLE_RESIDENT` (and symmetric for Critic), with
all `ERROR_RECOVERY` transitions cleanly reverting to `IDLE_RESIDENT`.

## Layout

```
inference/src/krisna_inference/
├── orchestrator/        # coordination layer — no GPU code
│   ├── swap_orchestrator.py    # the state machine driving load/unload/swap
│   ├── state_machine.py         # ResidencyState transitions (§7.4)
│   ├── model_registry.py         # ModelSpec/Tier/REGISTRY/LOW_VRAM_REGISTRY
│   ├── vram_budget.py             # VRAMLedger + RAMLedger
│   ├── design_state.py             # §5.1 Design State object
│   ├── flows.py                     # §5.3 sequence-flow helpers, offline-testable
│   ├── service.py                    # FastAPI app
│   ├── store.py                       # Design State persistence
│   ├── exceptions.py, logging_setup.py
├── backends/              # real model-loading implementations
│   ├── factory.py            # real_backend_factory — the one place env
│   │                            vars turn into constructed backend objects
│   ├── planner_backend.py     # Qwen3.5-9B, FROZEN, RAG + JSON-delta decode
│   ├── planner_rag.py          # TF-IDF retrieval over data-forge's UICrit corpus
│   ├── polish_default_backend.py  # Z-Image-Turbo, TRAINED (LoRA)
│   ├── polish_quality_backend.py   # Qwen-Image-Edit-2511, FROZEN
│   ├── critic_backend.py, critic_worker.py  # Gemma 4, FROZEN, isolated subprocess
│   ├── sketch_backend.py, sketch_handoff.py, maskgit_model.py
│   └── common.py, blob_store_singleton.py
└── verifiers/              # CLIP/SigLIP, OCR, layout-IoU, aesthetic/safety,
                              # handoff-consistency — §5's verifier stack
```

## Complete walkthrough — running a real agentic session

**1. Safe default — no GPU, no trained checkpoints:**
```bash
./scripts/inference/setup_env.sh
./scripts/inference/run_service.sh
# in another terminal:
python inference/runtime/run_agentic_session.py --finalize --critique
# or: krisna-session --finalize --critique
```
This exercises the full conversational-turn → finalize → critique
sequence against `MockBackend` — real orchestration logic (VRAM/RAM
ledger admission, mandatory swap exclusivity, OOM fallback), scripted
model output.

**2. Real backends, once you've trained something** (see
`../training/README.md`):
```bash
./scripts/inference/setup_env_inference.sh   # Planner/Sketch/both Polish tiers
./scripts/training/setup_env_critic.sh        # Critic's isolated venv (see below)

export KRISNA_USE_REAL_BACKENDS=1
export KRISNA_SKETCH_CHECKPOINT=./models/sketch_tier/checkpoint_final
export KRISNA_POLISH_DEFAULT_LORA_PATH=./models/polish_default_lora
export KRISNA_PLANNER_RAG_CORPUS_DIR=./models/planner_rag_index   # or data-forge's model_data/ directly

./scripts/inference/run_service.sh
# or, for a scriptable smoke test instead of the full service:
python inference/runtime/run_agentic_session.py --real --finalize --critique
```

**3. Low-VRAM mode**, if your GPU is 12-16GB instead of 24GB — see the
dedicated section below.

## Setup (Linux)

```bash
git clone <this-repo> krisna   # or unzip
cd krisna
chmod +x scripts/*/*.sh   # if the executable bit didn't survive the zip
./scripts/inference/setup_env.sh
```

## Run tests

```bash
./scripts/inference/run_tests.sh
```

## Run the service

```bash
./scripts/inference/run_service.sh
# defaults to http://127.0.0.1:8420 — override with:
# ./scripts/inference/run_service.sh 0.0.0.0 8420
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness + residency state + VRAM snapshot |
| POST | `/session` | create a session (§5.1 Design State) |
| GET  | `/session/{id}` | fetch current Design State |
| GET  | `/session/{id}/render` | serves finalized image render as raw PNG bytes |
| POST | `/session/{id}/message` | §5.3 conversational turn (forwards multi-turn history & prior critique) |
| POST | `/session/{id}/finalize` | §5.3 finalize (`{"quality": true}` for Qwen-Image-Edit-2511, default Z-Image-Turbo; auto-synthesizes rich prompt from intent) |
| POST | `/session/{id}/critique` | §5.3 critique pass (Gemma 4 31B). Body: `{"compare_against_image_ref": ..., "compare_against_score": ...}` (both optional — builds a DPO pair when given) |
| GET  | `/orchestrator/status` | residency state + VRAM + whether conversation is currently available |
| GET  | `/preference-pairs/stats` | pair counts by source |
| POST | `/preference-pairs/export` | writes a DPO JSONL export, optional `?source=` filter |

### Agentic Multi-Turn Pipeline Synchronization (§5.3)

Krisna's conversational pipeline connects all tiers in a closed multi-turn feedback loop:
1. **Multi-Turn Context Forwarding**: Every conversational turn forwards prior dialog history (`[turn.model_dump() for turn in state.conversation_history]`) into `PlannerBackend.run()`. Turns 2+ maintain full conversational memory.
2. **Structured Constraint Delta Merging**: Structured JSON updates emitted by the Planner (`style`, `palette`, `layout_hints`, `locked_regions`) are parsed and merged directly into `state.constraints`.
3. **Critic Feedback Loop Closure**: Following a critique pass, `state.critique.result` is passed into `orchestrator.run_conversational_turn(prior_critique=...)`. The Planner formats critique scores, dimension evaluations, and suggested edits directly into its prompt.
4. **CLIP Text Conditioning Grounding**: `SketchBackend` enriches the prompt with active design constraints (`message (style; palette; layout)`), preventing conditioning drift.
5. **Finalize Prompt Synthesis**: When `prompt` is not explicitly provided, `flows.finalize` derives an enriched prompt from conversation history and active constraints for both Polish and Verifiers.

Example session, end to end:

```bash
BASE=http://127.0.0.1:8420

SID=$(curl -s -X POST $BASE/session -H "Content-Type: application/json" \
  -d '{"style":"minimalist"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl -s -X POST $BASE/session/$SID/message -H "Content-Type: application/json" \
  -d '{"message":"design a login screen"}'

curl -s -X POST $BASE/session/$SID/finalize -H "Content-Type: application/json" \
  -d '{"quality": false}'

curl -s -X POST $BASE/session/$SID/critique

# Fetch the rendered image bytes directly
curl -s $BASE/session/$SID/render -o render.png
```

## OOM / VRAM behavior, concretely

- `POLISH_QUALITY` (Qwen-Image-Edit-2511, 16GB) has a configured
  `fallback_tier` of `POLISH_DEFAULT` (Z-Image-Turbo, 8GB) — per §6's
  stack table. If it OOMs (or a budget preflight shows it won't fit),
  the orchestrator automatically retries on the fallback and reports
  `degraded: true` in the result rather than silently substituting.
- `CRITIC` (Gemma 4 31B) has **no** fallback tier — there's only one
  critic in the stack. If it exhausts retries, the request fails
  (`OOMRecoveryExhausted`), but baseline residency (Planner + Sketch) is
  always restored before control returns, so the session stays usable.
- Restoring baseline after a failed swap gets more retry attempts than a
  normal load — it's the one operation this design has no fallback for.

## Low-VRAM / CPU-offload mode (12-16GB GPU, rest in system RAM)

**Yes, this is possible, and it's real — not a resize of the numbers.**
Set `KRISNA_LOW_VRAM_MODE=1` and construct the orchestrator with
`low_vram=True`:

```bash
export KRISNA_LOW_VRAM_MODE=1
export KRISNA_USE_REAL_BACKENDS=1
./scripts/inference/run_service.sh   # reads KRISNA_LOW_VRAM_MODE and constructs
                             # SwapOrchestrator(low_vram=True, envelope_gb=12.0)
```

Two genuinely different offload mechanisms are used, chosen deliberately
after checking what's actually documented to work (not assumed):

1. **Diffusion pipelines (Polish Quality — Qwen-Image-Edit-2511)** use
   diffusers' `enable_model_cpu_offload()`. **Not**
   `enable_sequential_cpu_offload()` — that has a **confirmed, documented
   incompatibility** with bnb NF4 quantization (diffusers GH issue #10800:
   `"Blockwise quantization only supports 16/32-bit floats, but got
   torch.uint8"`). `enable_model_cpu_offload()` moves whole submodules
   (text encoder, VAE, transformer) between GPU/CPU, keeping their
   already-quantized dtype intact while idle — the RAM cost here is close
   to 1:1 with the VRAM saved, not inflated.

2. **The Critic (Gemma 4)** bypasses unsloth's `FastModel` entirely in
   this mode — checked and confirmed unsloth has **no documented
   max_memory/CPU-offload support** (its own bug tracker shows users
   hitting plain OOM rather than a graceful offload path). Falls back to
   plain `transformers` + `bitsandbytes` with `max_memory={0: "...GiB",
   "cpu": "...GiB"}` + `llm_int8_enable_fp32_cpu_offload=True` — the
   real, documented pattern bitsandbytes requires for this
   (`max_memory` alone raises a clear error without that flag). This is
   correct but slower than unsloth's fast path — a real trade-off of
   this mode, not a bug.

**The important, easy-to-miss cost this second mechanism has**:
bitsandbytes stores the CPU-offloaded portion in **FP32, not 4-bit** — an
8x per-parameter size increase relative to NF4. For Gemma 4 31B Dense
(~30.7B params), offloading enough to bring GPU residency down to
~11.5GB (deliberately kept below the 12GB target card for real headroom
— see `model_registry.py`'s `LOW_VRAM_REGISTRY[Tier.CRITIC]` comment)
means roughly a third of the model's parameters living on the CPU side
at FP32: **~11.1B params x 4 bytes ≈ 45GB of system RAM** — not a small
number, and not proportional to the VRAM saved. This is a real, computed
lower bound from bitsandbytes' own documented behavior, not a guess — see
`model_registry.py`'s `LOW_VRAM_REGISTRY` comment for the full math. The
diffusion-pipeline offload above doesn't have this problem (no fp32
upcast), which is why `POLISH_QUALITY`'s RAM cost estimate (~10GB) is so
much smaller than `CRITIC`'s (~45GB) despite `CRITIC`'s smaller VRAM
target.

| Tier | Full-VRAM mode | Low-VRAM mode | Offload mechanism |
|---|---|---|---|
| Planner | 6.5GB / 0GB RAM | unchanged — already small | none, not needed |
| Sketch | 3.0GB / 0GB RAM | unchanged — already small | none, not needed |
| Polish Default (Z-Image-Turbo) | 14.0GB / 0GB RAM | 12.0GB / 2.0GB RAM | `enable_model_cpu_offload()` — a real but modest saving, since the DiT dominates this pipeline's weight (see the offloading-mechanism discussion above) |
| Polish Quality (Qwen-Image-Edit-2511) | 16.0GB / 0GB RAM | **~10GB / ~10GB RAM (estimate)** | `enable_model_cpu_offload()` |
| Critic (Gemma 4 31B) | 18.0GB / 0GB RAM | **11.5GB / 45.0GB RAM (computed, ~0.5GB headroom vs. a 12GB card — see docs/review/06 and 18)** | plain transformers+bnb, `max_memory` + `llm_int8_enable_fp32_cpu_offload` |

UPGRADE: this table previously showed Polish Default at its pre-fix
`8.0GB` (the registry's declared `quantization` for that tier used to be
wrong — see `docs/review/13_ram_offload_and_precision_audit.md` for the
full story) and Critic at its pre-headroom-fix `~12GB/~40GB` (zero
headroom against its own 12GB target — see `docs/review/06` and the
several review phases that found this exact fix reverting and reapplied
it). Both corrected here to match `model_registry.py`'s actual current
values, not re-described from memory.

Two configurable env vars for the Critic specifically:
`KRISNA_CRITIC_MAX_GPU_GB` (default `11.5` — deliberately kept below the
12GB target card rather than equal to it, to leave real headroom for
CUDA context/fragmentation/activation memory; must be changed together
with `LOW_VRAM_REGISTRY[Tier.CRITIC].vram_gb` in `model_registry.py` if
you override it), `KRISNA_CRITIC_MAX_CPU_GB`
(default `64`) — raise the CPU cap if your box has more RAM than the
default assumes, or lower the GPU cap further if 11.5GB is still too much
for your card (at the cost of an even larger RAM requirement, per the
math above).

**What's an estimate vs. what's verified:** the offload mechanisms
themselves (which diffusers/bitsandbytes APIs to call, and which
combination is safe) are verified against real documentation and known
GitHub issues, not guessed. The exact `vram_gb`/`ram_gb` numbers in
`LOW_VRAM_REGISTRY` are reasoned estimates (Critic's ~45GB figure is a
computed lower bound from documented behavior; Polish Quality's ~10GB/
~10GB is a rough estimate, not a benchmark) — validate against a real run
on your actual hardware and adjust the registry numbers to match what you
measure. `../tests/inference/test_low_vram_mode.py` checks the bookkeeping/wiring is
internally consistent; it cannot and does not claim to have measured real
GPU/RAM usage, since that requires an actual GPU this test environment
doesn't have.

## Planner (FROZEN — no training, real RAG retrieval instead)

**Under the final, no-RLHF-loop PRD revision, the Planner is never
fine-tuned.** `training/src/krisna_training/planner/` (BF16 LoRA
chat-SFT) still exists in this repo but is **deprecated** — it emits a
`DeprecationWarning` on import and is not called by anything in the
active pipeline. It's kept as a reference implementation in case a future
PRD revision un-freezes this tier; see that package's `__init__.py` for
exactly what would need to change to revive it.

Design-domain grounding instead comes from **real retrieval** over
data-forge's UICrit corpus:

- `inference/src/krisna_inference/backends/planner_rag.py` — a stdlib TF-IDF + cosine-similarity index
  over `model_data/planner_rag_corpus/uicrit_critiques.jsonl` (data-forge's
  real, human-annotated critique export — see data-forge's
  `s01_5_uicrit_join.py`/`s12_model_data_export.py`). Built once at
  `PlannerBackend.load()` time; queried per conversational turn.
- `inference/src/krisna_inference/backends/planner_backend.py` — folds the top-k retrieved critique
  snippets into the system prompt, then uses a **generate-validate-retry**
  loop (not a hard grammar constraint — no verified constrained-decoding
  support was found for this tokenizer/attention combination) to get a
  JSON design-state delta matching `design_state.py`'s schema. A malformed
  response after `MAX_JSON_RETRIES` raises `PlannerJSONDecodeError`, a
  distinct exception `flows.py` can handle differently from an OOM or load
  failure.

```bash
export KRISNA_PLANNER_RAG_CORPUS_DIR=/path/to/data-forge/model_data
export KRISNA_USE_REAL_BACKENDS=1
./scripts/inference/run_service.sh
```

If `KRISNA_PLANNER_RAG_CORPUS_DIR` isn't set, or the corpus file doesn't
exist yet, the Planner degrades cleanly to an empty retrieval index (a
warning is logged, not a crash) — the model still runs, just without
grounding context for that turn.

| Piece | What it is |
|---|---|
| `planner_rag.py::UICritRAGIndex` | TF-IDF index, built from data-forge's real critique corpus. `retrieve(query, k)` returns the top-k most relevant real human critiques. |
| `planner_backend.py::_extract_json_delta` | Finds the last balanced `{...}` in the model's output, validates it has the required `stage` key against the five valid `SessionStage` values, and fills sane defaults for the optional keys. Pure logic, tested without a real model in `../tests/inference/test_planner_json_delta.py`. |

`inference/src/krisna_inference/backends/` implements `TorchBackend`-equivalent
classes for every tier, wired into the same `ModelBackend` interface
`MockBackend` satisfies — the orchestrator, state machine, and tests don't
change at all when you switch backends.

| Tier | Backend | Model | Status |
|---|---|---|---|
| Planner | `planner_backend.PlannerBackend` | Qwen3.5-9B / Qwen3.5-4B (fast mode) | Real, **frozen** — RAG retrieval, no LoRA |
| Sketch | `sketch_backend.SketchBackend` | MaskGIT-lineage, checkpoint-pluggable | Architecture + inference-time sampler + real training code (see "Sketch tier training" below). No trained checkpoint ships with this repo — you train one. |
| Polish (default) | `polish_default_backend.ZImageTurboBackend` | Z-Image-Turbo | Real, **fine-tuned** — the one renderer this project actually trains (LoRA + Diffusion-DPO) |
| Polish (quality) | `polish_quality_backend.QwenImageEditBackend` | Qwen-Image-Edit-2511 | Real, **frozen** — zero-shot ICL edit conditioning + SDEdit-style partial denoising (`edit_strength`), NF4, no LoRA |
| Critic | `critic_backend.CriticBackend` | Gemma 4 31B Dense | Real, **frozen** on-demand product feature, runs in an isolated subprocess — see below |

### Enabling real backends

```bash
./scripts/inference/setup_env_inference.sh   # planner + sketch + both polish tiers, into .venv
export KRISNA_USE_REAL_BACKENDS=1
./scripts/inference/run_service.sh
```

Optional env vars: `KRISNA_PLANNER_FAST_MODE=1` (Qwen3.5-4B instead of 9B),
`KRISNA_PLANNER_RAG_CORPUS_DIR` (data-forge's `model_data/` directory —
see "Planner" above), `KRISNA_SKETCH_CHECKPOINT`,
`KRISNA_POLISH_DEFAULT_LORA_PATH` (Z-Image-Turbo's fine-tuned adapter —
the one tier that still has one), `KRISNA_POLISH_QUALITY_EDIT_STRENGTH`
(default `0.65` — SDEdit-style partial-denoising strength for
Qwen-Image-Edit-2511, **not independently confirmed as a real kwarg on
`QwenImageEditPlusPipeline` specifically** — see `polish_quality_backend.py`'s
inline comment before relying on this in production), `KRISNA_BLOB_ROOT`
(where generated images are written, default `./krisna_blobs`).

`KRISNA_PLANNER_LORA_PATH`, `KRISNA_CRITIC_LORA_PATH`, and
`KRISNA_POLISH_QUALITY_LORA_PATH` **no longer exist** — those three tiers
ship frozen. `tests/test_inference_factory.py::TestFrozenModelsHaveNoLoraWiring`
checks this directly against `factory.py`'s source, not just behavior.

### Critic tier isolation — read this before wiring it up

Gemma 4 31B (via Unsloth) needs `transformers==5.5.0` pinned **exactly**
(`unsloth_zoo` caps `transformers<=5.5.0`, Gemma 4 itself needs `>=5.5.0`).
The Planner tier (Qwen3.5) needs `transformers` built from git main, which
is newer than 5.5.0 and not interchangeable with it. **These two pins
cannot both be satisfied in one venv.**

Rather than silently picking one pin and quietly breaking the other tier,
the Critic tier runs `critic_worker.py` as a **separate subprocess in its
own venv**, talking to the main process over stdin/stdout JSON lines — the
same pattern already used for vLLM subprocessing in the data-forge
project's `engine.py`. Note this isolation reasoning is about
`transformers` version pins, not about training — it applies whether or
not this tier is ever fine-tuned, which is why it's unaffected by the
freeze.

```bash
./scripts/training/setup_env_critic.sh   # creates ./venv-critic, isolated deps
export KRISNA_USE_REAL_BACKENDS=1
# KRISNA_CRITIC_VENV_PYTHON defaults to ./venv-critic/bin/python
./scripts/inference/run_service.sh
```

If `./venv-critic` doesn't exist, `CriticBackend.load()` fails immediately
with a clear message rather than a confusing subprocess crash — this is
exactly the OOM-recovery-adjacent path `swap_orchestrator.py` already
handles: a critique request fails cleanly and baseline residency is
restored, the system doesn't get stuck.

Gemma 4's critique output uses `critique_source: "gemma4_31b_frozen"`
(renamed from `"gemma4_31b_qlora"`, which implied a trained adapter that
never gets applied under the final PRD) — see `critic_worker.py::CRITIQUE_SOURCE`.

### What's genuinely still missing

- **A real VRAM/dtype consistency bug was found and fixed reviewing this
  against `model_registry.py`'s declared budget**: `PlannerBackend`
  defaulted to unquantized BF16 (~18GB for a 9B model) while
  `model_registry.py` declares this `always_resident=True` tier at
  `vram_gb=6.5` (a 4-bit-scale number). That mismatch would have blown
  PRD §5.3's "Tier A must never exceed ~10GB resident" rule before the
  sketch tier or anything else could load. Now defaults to NF4, matching
  the declared budget — see `planner_backend.py`'s module docstring and
  `../tests/inference/test_planner_vram_budget_consistency.py`, which checks the
  actual idle-state VRAM sum against the envelope directly rather than
  just asserting the two numbers individually.

- **VQ-decode → pixel → renderer-latent handoff is now real** (see
  `inference/src/krisna_inference/backends/sketch_handoff.py` + the sketch training package's
  `vq_tokenizer.py`).
- **Region-locking as a real pipeline parameter.** `QwenImageEditBackend`
  currently expresses `constraints.locked_regions` as a prompt-level
  instruction rather than a genuine differential-diffusion mask parameter
  — diffusers' locked-region API for `QwenImageEditPlusPipeline` was still
  moving as of this build; tighten this once it stabilizes.
- **`KRISNA_POLISH_QUALITY_EDIT_STRENGTH`'s `strength` kwarg is unverified
  for `QwenImageEditPlusPipeline` specifically** (confirmed only for
  `QwenImageImg2ImgPipeline`/`QwenImageInpaintPipeline` — architecturally
  different pipeline classes). Confirm against the installed diffusers
  version's actual `__call__` signature before depending on this control
  in production; if wrong, it raises a clear `TypeError` at call time
  rather than silently no-op'ing.
- **Training code only exists for the two tiers this project actually
  trains** — the sketch tier (from scratch) and Z-Image-Turbo (LoRA +
  Diffusion-DPO). `training/planner/` and `training/critic/` are real,
  working code but deliberately deprecated (see their `__init__.py`
  docstrings) — the Planner and Critic ship frozen under the final PRD.
  There is no DPO trainer consuming `../training/src/krisna_training/dpo/dpo_dataset_export.py`'s output
  either — that module builds and exports preference pairs, it doesn't
  train on them.

## Verifier stack + Critique Adapter + DPO pipeline

`inference/src/krisna_inference/verifiers/` and `training/src/krisna_training/dpo/`
implement §6's verifier stack ("CLIP/SigLIP, OCR, layout-IoU, aesthetic/
safety, handoff-consistency. <1B combined, off-the-shelf") and the DPO
preference-pair pipeline it feeds.

```bash
./scripts/setup_env_verifiers.sh   # shares the main venv — no version conflict
```

| Verifier | Real model | Grounding |
|---|---|---|
| `clip_alignment` | CLIP ViT-L/14 (text-image cosine sim) | Standard, well-known |
| `ocr_readability` | easyocr | Per-region confidence + legibility heuristic |
| `layout_iou` | OpenCV contour detection (default, pluggable) | Real but not a trained UI-element detector — see module docstring |
| `aesthetic` | LAION-AI/aesthetic-predictor (linear head on CLIP ViT-L/14) | github.com/LAION-AI/aesthetic-predictor |
| `handoff_consistency` | CLIP ViT-L/14 (image-image cosine sim) | Reuses the same embedder as `clip_alignment` |
| `safety` (gate, not a scorecard field) | Falconsai/nsfw_image_detection | ~80M HF downloads, binary normal/nsfw ViT |

`verifier_stack.VerifierStack.score_finalize_output()` returns a dict with
exactly `design_state.VerifierScores`' field names — `flows.finalize()`
assigns it straight into `finalize_output.verifier_scores` when you pass a
`VerifierStack` instance (the packaged service does this automatically
when `KRISNA_USE_REAL_BACKENDS=1`).

### Critique Adapter (§5.2, formalized)

`src/krisna_inference/verifiers/critique_adapter.py` has two entry points, both producing the
exact same `CritiqueResult` shape:
- `from_gemma_output(raw)` — re-validates whatever `critic_worker.py`'s
  subprocess returned through the same pydantic model every other source
  goes through.
- `from_verifier_scores(scores)` — a cheap proxy critique built entirely
  from the verifier stack, no Critic-tier swap needed. Only 4 of the 5
  verifier scores map onto a critique dimension (`handoff_consistency` has
  no §5.2 dimension counterpart) — documented as an honest proxy, not
  presented as equivalent to a real Gemma judgment.

### DPO preference pairs

`../training/src/krisna_training/dpo/preference_store.py` is a SQLite table of `(chosen_ref, rejected_ref,
prompt, source)` rows. Three ways pairs get built:

1. **`pair_builder.build_pair_from_candidates()`** — ranks any set of
   scored candidates (e.g. Z-Image-Turbo vs. Qwen-Image-Edit-2511 output
   for the same prompt) and pairs best vs. worst, skipping near-ties
   (`min_score_gap`, default 0.1) since those aren't a real preference
   signal.
2. **`flows.critique_pass(..., preference_store=..., compare_against=...)`**
   — wires a Gemma critique into a pair when the caller supplies a second
   scored candidate. `DesignState` only tracks one current
   `finalize_output` (§5.1), not a history, so this needs that second
   candidate explicit rather than pretending automatic cross-session
   history tracking exists.
3. **`dpo/uicrit_importer.import_records()`** — seeds pairs from UICrit
   human ratings (§6: "seeded by UICrit..."). **Field names are NOT
   verified against UICrit's actual schema** — I have not personally
   inspected its real CSV/JSON columns, so the importer takes plain dicts
   plus field-name overrides rather than guessing wrong names silently.
   Check the real dataset before wiring this up for real.

`../training/src/krisna_training/dpo/dpo_dataset_export.py` writes a standard `{"prompt", "chosen",
"rejected"}` JSONL, the shape `trl`'s `DPOTrainer` and most diffusion-DPO
scripts expect — or hit `/preference-pairs/export` on the running service.
The actual DPO training loop is a separate build, out of scope here.

## Production Docker Containerization & Dual-Venv Isolation

For release and production deployments, Krisna Inference is containerized with strict dependency isolation:

```bash
# Launch multi-service GPU stack (FastAPI Inference on 8420 + Web Studio on 3000)
docker compose up -d

# Or using host preflight launchers:
./scripts/docker/run_docker.sh --build                     # Linux / macOS
.\scripts\docker\run_docker.ps1 -Build                    # Windows native PowerShell
```

### Multi-Environment Isolation Architecture
- `/opt/venv-inference`: Main tier virtualenv running `torch>=2.6.0`, `transformers>=5.2.0`, `diffusers>=0.31.0`, SwapOrchestrator, Planner (Qwen3.5), Sketch (MaskGIT), and Polish models.
- `/opt/venv-critic`: Isolated Critic tier virtualenv running `transformers==5.5.0` + `unsloth` + `unsloth_zoo` for Gemma 4 31B Dense.
- Out-of-process communication: `CriticBackend` talks to `critic_worker.py` over stdin/stdout JSON lines without Python symbol collisions.
- Hardware fail-fast diagnostics: `python scripts/inference/check_hardware.py --require-gpu` validates NVIDIA GPU presence, CUDA driver, compute capability, and VRAM before launching. MockBackend is strictly reserved for CI and automated testing.


