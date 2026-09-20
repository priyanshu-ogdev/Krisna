# Phase 12 — Post-Upgrade Resync Audit (data-forge fix, inference-frontend, installer, SOTA re-check)

This phase covers everything that changed in the follow-up working
session after Phases 1–11 were written, plus a fresh SOTA/citation
re-check against sources published since those phases (all confirmed via
live web search on 2026-09-15, not from training-data memory — model
names/dates below are stated as fact only where a primary or clearly
authoritative secondary source confirmed them this session).

Method, same as every other phase in this review: read the actual code,
cross-check claims against it, flag anything genuinely out of sync,
cite primary sources.

---

## §1 — `maskgit_vq` dead-stage removal (closes Phase 1 & Phase 5 findings)

**What Phase 1 flagged, unresolved at the time:** `data-forge`'s
`maskgit_vq` (Open-MAGVIT2) encoder had no working `.encode()`/`.decode()`
implementation — its loader in `engine.py` always raised `RuntimeError`
by design. `s08_encoding.py`'s VQ-token branch caught that failure
silently, so it ran, failed, and logged a warning on every `ui_first`
record, for an artifact nothing downstream read (the real training path,
`data_forge_bridge/sync_sketch_tier.py`, re-tokenizes raw images through
the actual tokenizer, `boris/vqgan_f16_16384`, instead).

**What digging into option (a) — delete the dead stage — surfaced that
Phase 1 did not catch:** the silent failure fed directly into
`utils/completeness.py`'s `is_encoding_complete()`, which required
`vq_tokens` for every `ui_first` record. Since that artifact was never
producible, this predicate was **False for every `ui_first` record, on
every run, from the start**. Two real consumers depend on it:

- `s09_heldout.py` excludes any record failing `is_encoding_complete()`
  from the training pool with reason `"encoding_incomplete"` — so the
  entire `ui_first` domain was silently carved into heldout on every
  run.
- `s12_model_data_export.py`'s `_export_sketch_tier` filters on
  `r.domain == "ui_first" and is_encoding_complete(r)` — so the Sketch
  tier's exported training folder was **empty on every run**.

This is a severity upgrade from Phase 1's own framing ("dead weight...
functionally worked around") to a live, pipeline-wide data-loss bug. It
was invisible in Phase 1's review because that phase checked whether the
*training* path was sound (it is — the bridge never depended on
data-forge's VQ tokens) without also tracing what `is_encoding_complete`
gated downstream.

**Fix applied (all changes, verified against the full `data_forge` test
suite — 114 passed):**
- `data-forge/configs/models.yaml` — `maskgit_vq` entry deleted.
- `data-forge/data_forge/inference/engine.py` — the always-failing loader
  branch deleted.
- `data-forge/data_forge/stages/s08_encoding.py` — the dead encode
  branch deleted.
- `data-forge/data_forge/utils/completeness.py` — `vq_tokens` dropped
  from `_REQUIRED_UI_FIRST_ONLY`; `ui_first` and `general_design` now
  require the identical artifact set (`z_image_latent`, `control_map`).
  **This is the actual bug fix** — everything else is cleanup.
- `s12_model_data_export.py`, `s09_heldout.py`, `config.py` — dead
  `vq_tokens/` linking code, an unused path constant, and stale comments
  removed/corrected.
- Tests updated to assert the corrected (not the old, broken) behavior.

**Status: Resolved.** Phase 1 and Phase 5 findings above have been
annotated in place pointing here.

---

## §2 — `vram_budget.py` safety-margin doc bug (closes Phase 6 finding)

Phase 6 flagged, as "Low" severity, that the Critic tier's low-VRAM
sizing (`vram_gb=12.0` against a `12.0`GB envelope) fits with exactly
zero headroom. Re-reading `vram_budget.py` in this session found that
its own docstring claimed this was already mitigated —
`_BaseLedger.safety_margin_gb`'s comment stated *"VRAMLedger sets a
non-zero default."*

**That claim was false.** Grepped the entire `inference/` tree for
`safety_margin_gb`: `VRAMLedger` sets no such override, and
`SwapOrchestrator.vram_safety_margin_gb` also defaults to `0.0` —
deliberately, per that class's own (correct) docstring, because a
nonzero default would make the zero-headroom Critic tier permanently
inadmissible (it has no `fallback_tier`). So the real, intentional
design was: `0.0` default, explicit opt-in for operators who want
margin on real hardware. Only the comment was wrong, claiming a
protection that doesn't exist.

**Fix applied:** the docstring in `vram_budget.py` now states the actual
design and the actual reason for it, matching `swap_orchestrator.py`'s
correct version instead of contradicting it.

**Status: Resolved** — documentation-accuracy bug, not a behavior bug.
The zero-headroom sizing itself remains a legitimate "Low" severity item
per Phase 6 (an operator on real hardware may still want to pass
`vram_safety_margin_gb` explicitly) — that recommendation stands
unchanged.

---

## §3 — `scripts/inference/download_weights.py` — new artifact, path claims verified against real training code

This is new infrastructure (didn't exist at Phase 1–11 time), so it gets
its own findings rather than annotating an existing phase.

**What it does:** downloads the four frozen HF-hosted models
(Planner, Polish-Default, Polish-Quality, Critic) and locates this
repo's own trained artifacts (Sketch checkpoint, Polish LoRA).

**Finding (self-caught and fixed before shipping, documented here for
the audit trail):** the first version's local-artifact auto-discovery
globs (`training/sketch_runs/**/final.pt`,
`training/polish_runs/**/adapter_model.safetensors`) were written from
assumption, not from reading the actual save code — and were wrong on
every count. Traced the real save paths instead:

| Artifact | Real save call | Real path (from actual configs) |
|---|---|---|
| Sketch checkpoint | `sketch/checkpoint_io.py`'s `save_checkpoint()`, called from `sketch/train.py` | `<output_dir>/checkpoint_final.pt`, where `output_dir` = `./checkpoints/sketch_stage{1,2}_*` per `configs/sketch_train_stage{1,2}_*.yaml`, relative to the repo root (confirmed: `scripts/training/train_sketch_stage*.sh` runs `python -m krisna_training.sketch.train` from there, no `cd` first) |
| Polish LoRA (dreambooth base fine-tune) | diffusers' own `train_dreambooth_lora_z_image.py`, wrapped by `scripts/training/train_polish_default_lora.sh` | `<output_dir>/` directly (no subfolder) — `./checkpoints/polish_default_lora/` per `configs/polish_default_lora_z_image.yaml`. Confirmed against that script's own printed instruction: `export KRISNA_POLISH_DEFAULT_LORA_PATH=<output_dir>` |
| Polish LoRA (DPO-refined, preferred when present) | `polish/train_dpo.py`, wrapped by `scripts/training/train_polish_dpo.sh` | `<output-dir>/final/` — `models/dpo_checkpoints/stage1_general/final/` per `configs/dpo_z_image_stage1_general.yaml`. Confirmed against that script's own printed instruction: `export KRISNA_POLISH_DEFAULT_LORA_PATH=<output-dir>/final` |

Both possible Polish-LoRA shapes are now globbed for, most-recently-
modified wins — which naturally prefers the DPO-refined adapter when
both exist, without needing to special-case which script ran last.

**Verification:** created artifacts at the corrected real paths and
confirmed `download_weights.py --dry-run` auto-discovers both
(`local_artifact_found` events for `SKETCH_CHECKPOINT` and
`POLISH_LORA`, `dry_run_complete` with an empty `would_fail` list).

**Status: Resolved**, verified by direct test rather than by inspection
alone.

---

## §4 — `inference-frontend/` (Node control panel) — sync against the real wire contracts

New artifact, not covered by any earlier phase. Reviewed for the one
failure mode that matters for a UI layer: does it correctly reflect what
the Python backend actually returns.

**Finding (caught and fixed):** the Studio chat renderer initially read
`turn.message` / `turn.text`. The actual wire schema
(`orchestrator/design_state.py`'s `DesignState.to_wire()`) returns
`conversation_history: list[{role, content, timestamp}]`. Fixed to read
`conversation_history[].content`.

**Verified by live integration test**, not just code reading: ran the
real FastAPI service (`uvicorn krisna_inference.orchestrator.service:app`)
with `MockBackend` (deterministic, no GPU) behind the Node proxy end to
end — session create → message → finalize → critique — and confirmed
every field the frontend renders (`stage`, `conversation_history`,
`finalize_output.verifier_scores`, `critique.result`,
`/orchestrator/status`'s `vram.{envelope_gb,used_gb,resident}`,
`low_vram_mode`, `residency_state`) is present and correctly shaped in
the real response.

**Also verified against source, not assumed:** the GPU-memory-rack
visualization's per-tier VRAM figures were checked line-by-line against
`model_registry.py`'s actual `REGISTRY`/`LOW_VRAM_REGISTRY`, and the
residency-state diagram's states/edges were checked line-by-line against
`state_machine.py`'s actual `TRANSITIONS` table. Both matched exactly.

**Real-backend wiring verified negatively (which is the correct proof
for "no mock"):** ran the FastAPI service with
`KRISNA_USE_REAL_BACKENDS=1` forced (as `BackendProcess.start()` in
`server.js` now always does) in this sandbox, which has no GPU/`torch`
installed. It correctly attempted to load the real
`Qwen/Qwen3.5-9B` planner and failed loudly —
`BackendLoadError: Failed to load planner 'Qwen/Qwen3.5-9B': No module
named 'torch'` — rather than silently falling back to MockBackend. That
failure, in this environment, is the evidence the real path is genuinely
wired and not quietly substituted.

**Status: Resolved.**

---

## §5 — SOTA re-check: model identities and specs, verified against sources published since Phases 1–4

Phases 3 and 4 cited Qwen3.5-9B and Gemma 4 31B Dense but Phase 4
explicitly flagged: *"needs a specific Gemma-4 primary source before
this goes in the paper; not independently verified in this review."*
Re-checked both this session via live search.

### Qwen3.5-9B (Planner) — confirmed real, citation stands
Released 2026-03-02 as part of the Qwen3.5 Small Model Series (0.8B/
2B/4B/9B), alongside the 2026-02-16/02-24 releases of the larger
397B-A17B/122B-A10B/35B-A3B/27B tiers. Confirms Phase 3's architecture
claim: hybrid linear-attention (Gated DeltaNet) + Gated Attention,
native multimodal (unified early-fusion text+vision training, not an
adapter/bridge). No correction needed to Phase 3's citations (#15, #16).

- **New citation to add:** Qwen Team, Alibaba (2026-03-02). *Qwen3.5
  Small Model Series (0.8B/2B/4B/9B) release notes.* Hugging Face /
  ModelScope / `github.com/QwenLM/Qwen3.8` release log. (The GitHub repo
  itself is versioned `Qwen3.8` post-hoc, reflecting the family's later
  naming — cite the dated release-note entry, not the repo name, for the
  9B model specifically.)

### Z-Image-Turbo (Polish-Default) — confirmed real, and a genuine new finding on top
Released 2025-11-27 by Alibaba's Tongyi-MAI team. Confirms Phase 2's
identification. New, useful-for-the-paper technical detail Phase 2
didn't have: the actual distillation method is **Decoupled-DMD**
(Distribution Matching Distillation, with CFG-augmentation and
distribution-matching mechanisms deliberately separated — the former as
the quality "engine," the latter as the anti-mode-collapse "shield") plus
**DMDR** (fusing DMD with reinforcement learning), achieving 8-step
(8-NFE) generation. Architecture is **S3-DiT** (Scalable Single-Stream
DiT — text, visual-semantic, and VAE tokens concatenated as one sequence,
not a dual-stream design). 6B parameters.

- **New citation to add:** Tongyi-MAI, Alibaba (2025). *Z-Image: An
  Efficient Image Generation Foundation Model with Single-Stream
  Diffusion Transformer.* Technical report:
  `github.com/Tongyi-MAI/Z-Image/blob/main/Z_Image_Report.pdf`. This is
  the primary source for Decoupled-DMD/DMDR and S3-DiT — cite this
  directly rather than secondary coverage for anything architecture- or
  distillation-related in the paper.

**New finding (Medium severity) — declared VRAM for Polish-Default may
be under-sized:** `model_registry.py`'s full-mode `REGISTRY` declares
`Tier.POLISH_DEFAULT` at `vram_gb=8.0`. `polish_default_backend.py`
loads the model at `bfloat16` (confirmed by reading
`resolve_dtype(self.dtype)` with `dtype: str = "bfloat16"` as the
backend's own default) — at 2 bytes/param, a 6B-parameter transformer
alone is ≈12GB, before the text encoder(s), VAE, and LoRA adapter this
tier also loads. Published third-party reports (ComfyUI Wiki, 2025-11-27;
consistent across multiple independent sources checked this session) put
Z-Image-Turbo's practical VRAM floor at "under 16GB" for the *full*
pipeline. An `8.0`GB declared figure looks optimistic against both the
raw bf16 arithmetic and the model's own published consumer-GPU framing.

This matters beyond bookkeeping: `swap_orchestrator.py`'s admission
decisions are based on the **declared** ledger value, not a live probe
(by design, for determinism — see Phase 6). If `8.0` under-states the
real footprint, the orchestrator can admit Polish-Default into a VRAM
budget that doesn't actually fit it, risking a real CUDA OOM the ledger
exists specifically to prevent. **Recommend:** measure actual resident
VRAM for this tier on real hardware before the next training/inference
run that uses it, and correct `vram_gb` in `model_registry.py`
accordingly (both `REGISTRY` and `LOW_VRAM_REGISTRY`) — this review
cannot measure real GPU memory itself and is flagging the arithmetic
mismatch, not asserting a measured number.

### Gemma 4 31B Dense (Critic) — now confirmed, closes Phase 4's open gap
Gemma 4 released 2026-04, Google DeepMind, Apache 2.0 license (a
license change from Gemma 3's custom terms — itself worth noting for the
paper's licensing section, since it changes what commercial use is
permitted). Eight variants: E2B/E4B (edge, 2B/4B), 26B-A4B (MoE, 26B
total/4B active), and **31B dense** — confirms Phase 4's exact model
identification. Up to 256K context, natively multimodal (text/image/
video/audio on the smaller sizes).

- **New citation to add, replacing Phase 4's unresolved placeholder:**
  Google DeepMind (2026). *Gemma 4.* Official model page:
  `deepmind.google/models/gemma/gemma-4/`, and documentation:
  `ai.google.dev/gemma/docs`. Cite the official model page as primary;
  a from-scratch technical report/paper was not surfaced in this
  session's search and may not exist as a separate arXiv artifact the
  way Gemma 1–3 had — if the paper needs a peer-reviewed citation
  specifically, check for a Gemma 4 arXiv entry again closer to
  submission, since this family ships fast and a report may land after
  this review.

**Status: §5 closes Phase 4's open citation gap and upgrades Phase 2's
Z-Image-Turbo citation from secondary to primary-source-verified. One
new Medium-severity finding raised (Polish-Default VRAM sizing) — not
yet fixed, flagged for hardware verification.**

---

## §6 — SOTA re-check: is anything in this stack now behind current best-practice fast inference?

Scoped narrowly per this phase's mandate ("review other SOTA
implementations for fast and best inference") — checked whether the
project's chosen serving strategies for each tier are still reasonable,
not whether a totally different architecture would be better (that's a
much larger question than a sync audit answers).

- **Planner/Critic (LLM tiers):** both are loaded via plain
  `transformers` generation (confirmed: `planner_backend.py`,
  `critic_worker.py` both `import torch` +
  `AutoModelForCausalLM`-style loading, no vLLM/TGI/SGLang serving layer
  present anywhere in `inference/backends/`). For a **single-request,
  single-GPU, swap-heavy** serving pattern — this system loads/unloads
  whole tiers rather than serving concurrent requests to a resident
  model — a dedicated inference server (vLLM's continuous batching,
  PagedAttention) buys the most when serving *many concurrent requests*
  against one resident model. That's not this system's access pattern
  (`SwapOrchestrator` explicitly serializes: one tier resident for
  interactive use, others swapped out). Plain `transformers` generation
  is a reasonable, not-behind-SOTA choice for this specific access
  pattern; adopting vLLM here would add real complexity (a second
  serving process, KV-cache management) for a benefit (multi-request
  throughput) this system's design doesn't currently need. **Not a
  finding — a design choice worth stating explicitly in the paper's
  systems-design section, with the reasoning above, rather than left
  implicit.**
- **Polish-Default/Polish-Quality (diffusion tiers):** both go through
  `diffusers` pipelines. Z-Image-Turbo's own 8-NFE distillation (§5
  above) is itself the SOTA fast-inference technique for this model —
  there's no faster standard sampler to swap in underneath a model
  that's already distilled to 8 steps; further speedups would mean a
  different base model, not a different sampler. **Not a finding.**
- **Sketch tier (from-scratch MaskGIT):** already covered in Phase 1 —
  confirmed MaskGIT's parallel iterative-decoding inference procedure
  (not autoregressive) is implemented correctly, and Phase 1 already
  flagged the Halton-scheduler alternative (Besnier et al., 2025) as a
  drop-in option worth benchmarking, not yet adopted. That recommendation
  is unchanged and still open.

---

## Findings summary (this phase only)

| Severity | Finding | Status |
|---|---|---|
| **Resolved** | `maskgit_vq` dead stage caused silent, pipeline-wide Sketch-tier data loss (empty exports, all `ui_first` records wrongly held out) — far more severe than Phase 1's original framing | Fixed, 114/114 `data_forge` tests passing |
| **Resolved** | `vram_budget.py` docstring falsely claimed a nonzero safety-margin default existed | Fixed — comment now matches actual, correct design |
| **Resolved** | `download_weights.py`'s local-artifact auto-discovery paths were unverified assumptions, wrong on every count | Fixed, verified against real training save code + dry-run test |
| **Resolved** | Frontend chat renderer read the wrong field names for conversation history | Fixed, verified via live integration test against real backend |
| **Medium, open** | `model_registry.py`'s `POLISH_DEFAULT` `vram_gb=8.0` looks under-sized against bf16 arithmetic for a 6B-param model and the model's own published VRAM guidance | **Not fixed** — needs a real hardware measurement, flagged for the next run that has GPU access |
| **Closed (citation gap)** | Phase 4 lacked a verifiable Gemma-4 primary source | Closed — official DeepMind model page now cited |
| Info | Planner/Critic's plain-`transformers` serving (vs. vLLM/TGI) is the correct choice for this system's swap-heavy, single-resident-tier access pattern, not a gap — worth stating explicitly in the paper rather than leaving implicit | No action needed |

## Citations to add to `07_consolidated_citations.md`

24. Qwen Team, Alibaba (2026-03-02). *Qwen3.5 Small Model Series (0.8B/2B/4B/9B) release notes.* Hugging Face / ModelScope.
25. Tongyi-MAI, Alibaba (2025). *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.* Technical report, `github.com/Tongyi-MAI/Z-Image`.
26. Google DeepMind (2026). *Gemma 4.* Official model page, `deepmind.google/models/gemma/gemma-4/`.

---
Proceeds/supersedes: Phases 1, 2, 4, 5, 6 (annotated in place, not
rewritten — this file is the changelog). Phases 3, 7–11 unaffected,
re-confirmed accurate on spot-check, not re-derived from scratch in this
pass.
