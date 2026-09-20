# Phase 22 — Root Orchestration Verified End-to-End, VQGAN Installer Gap Closed, Stale Docs Fixed

Direct answer to this session's questions: **does the root tooling
orchestrate data-forge and training properly, is inference connected to
training output, and is the frontend actually wired to it** — yes, after
one real gap (VQGAN) was found and closed. Everything below was checked
against the actual code, not re-described from an existing doc or a
prior session's own summary.

---

## §1 — Confirmed already correct (spot-verified, not re-derived from scratch)

The prior session's own root-scripts work checked out on inspection:
`setup.sh`, `run_data_forge.sh`, `run_inference.sh`, `train.sh` all pass
`bash -n`, `train.sh --list`'s per-tier hyperparameter summaries match
the live YAML configs, and the Critic/Sketch/Polish fixes from earlier
phases (11.5GB/45GB headroom, CFG guidance, the `handoff_hook` wiring)
are all genuinely present in the code, not just claimed. Diffed this
session's upload against the prior merged package first: the only
substantive code change was one stale cross-reference comment in
`model_registry.py` (Polish-Default's zero-headroom situation used to
say "same as Critic below," which stopped being true once Critic's
headroom got fixed) — corrected, with the real distinction explained
(Critic's GPU target was an adjustable knob; Polish-Default's 12GB
figure is already "the DiT alone," nothing further to offload).

## §2 — Real gap found: the VQGAN decoder was never connected to the installer

Traced the full chain a fresh operator would actually follow:
`setup.sh` → `run_data_forge.sh` → `train.sh` → install-for-inference →
`run_inference.sh`. Every step connected cleanly except one: the VQGAN
decoder (`boris/vqgan_f16_16384`, downloaded by
`scripts/training/download_vqgan.sh`) is required by
`orchestrator/service.py`'s `VQTokenizer` construction — `KRISNA_VQGAN_
CHECKPOINT`/`KRISNA_VQGAN_CONFIG` env vars, with a real runtime error
(`"Cannot finalize: KRISNA_VQGAN_CHECKPOINT/KRISNA_VQGAN_CONFIG..."`) if
they're unset. `scripts/inference/download_weights.py` — the actual
inference installer, the thing `inference-frontend/`'s Setup tab drives
— had **zero** awareness this checkpoint existed. Nothing in the
documented setup flow ever told an operator to run `download_vqgan.sh`
separately or export those two variables; the first anyone would learn
about this requirement was a runtime crash on the first real `/finalize`
call.

**Fixed**: `download_weights.py` gained a VQGAN step — auto-discovers
`checkpoints/vqgan/{last.ckpt,model.yaml}` (matching `download_vqgan.sh`'s
own default destination) and, if not found, downloads it directly
(reimplementing that script's two `curl` calls via Python's stdlib
`urllib`, with the same JSON-event-per-line/retry contract every other
download in this script already uses — no new dependency). Writes
`KRISNA_VQGAN_CHECKPOINT`/`KRISNA_VQGAN_CONFIG` into `.env.inference`
alongside everything else. New `--vqgan-checkpoint`/`--vqgan-config`/
`--skip-vqgan` flags for anyone managing it a different way. Verified via
`--dry-run` that the new step is correctly sequenced into the existing
plan/disk-check/download flow (network access to the actual host isn't
available in this environment, so the live download path itself
inherits the same disclosed limitation as every other GPU/network-gated
claim in this review — the *logic* is exercised and correct, the actual
transfer isn't proven here).

## §3 — Claimed-but-not-persisted fixes, found again, reapplied for real

Consistent with this review's now well-established pattern: a prior
session's own summary claimed three shell-script "next step" hints had
been added (`train_sketch_stage1.sh`, `train_sketch_stage2.sh`,
`download_vqgan.sh`). Checked the actual files — none of the three were
there. Reapplied all three, each pointing at the exact real output path
and the exact env var that consumes it (verified against `config.py`,
`download_weights.py`'s own path-discovery globs, and `service.py`'s
VQTokenizer construction, not just written plausibly).

## §4 — A real, previously-undisclosed test-suite bug

Running `pytest tests/` from a clean checkout: **collection aborted
entirely** with `ModuleNotFoundError: No module named 'torch'` from
`tests/training/test_dpo_loss.py`. Every other torch-dependent test file
in this repo uses `torch = pytest.importorskip("torch")` specifically so
a missing torch install skips cleanly instead of failing the whole run
— this one file alone used a bare `import torch`. The root README's own
claim ("no GPU required... tests skip cleanly if torch is absent") was
therefore false for the repo as shipped: the *whole suite* wouldn't even
collect, let alone run, without torch installed. Fixed to match the
established pattern. Verified: 356 tests now collect cleanly, 346 pass
and 10 skip (torch-gated) with no torch present.

## §5 — Root README and `src/README.md` rewritten, both were stale

Root README still described `src/` as "the frontend, design deferred"
and never mentioned any of the four root orchestration scripts or
`inference-frontend/` at all — written before either existed. Rewrote it
with the verified end-to-end path from §§1–2 above, the real API routes,
and the real (corrected) test count/behavior from §4.

`src/README.md` itself listed endpoints that never existed under those
names (`POST /conversation/turn`, `POST /finalize`, `POST /critique` —
the real routes are session-scoped: `POST /session/{id}/message`,
`POST /session/{id}/finalize`, etc.) and described the frontend as
not-yet-designed when a full one already exists. Rewritten to point at
`inference-frontend/` and list the real, current routes rather than
carry forward a stale, wrong API surface.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, fixed — closes the training→inference connection gap** | `download_weights.py` had no knowledge of the VQGAN decoder `service.py` requires for every real Finalize call | Fixed — auto-discovery + auto-download added, verified via dry-run sequencing |
| **Real, fixed (recurrence)** | Three shell-script "next step" hints, claimed added in a prior session, were not actually present | Reapplied for real, verified on disk after editing |
| **Real, fixed** | `test_dpo_loss.py`'s bare `import torch` broke collection for the entire `pytest tests/` run without torch installed, contradicting the README's own claim | Fixed to match the established `importorskip` pattern; 356 tests now collect and run cleanly |
| **Real, fixed** | Root README and `src/README.md` both stale — no mention of the root scripts, `inference-frontend/`, or the real API surface; `src/README.md` listed endpoints that never existed | Both rewritten against verified, current code |
| Info | `model_registry.py`'s stale Polish-Default/Critic cross-reference comment | Confirmed already fixed on arrival this session (diffed against the prior package) |

## Answering the questions directly

- **Does the root tooling orchestrate the entire data-forge and
  training pipeline properly?** Yes — `setup.sh`/`run_data_forge.sh`/
  `train.sh` are real, syntax-clean, and each tier's launcher now
  correctly tells you what it produced and what to do next.
- **Can we proceed with inference after training? Is it connected?**
  Yes, now — the one real missing link (VQGAN) is closed. Every other
  training-output → inference-input path (Sketch checkpoint, Polish
  LoRA, both HF-frozen models) was already correctly connected.
- **Is the inference layer connected to the frontend?** Yes, verified
  in the prior merge session and re-confirmed here: `inference-frontend/`
  drives the real installer, starts the real backend, and its Studio
  tab is a working UI over the real, current API surface.
