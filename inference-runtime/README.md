# inference-runtime/

A CLI harness for running a real agentic session end-to-end against
checkpoints in `models/` — for manually testing trained models. Not a
production entry point (that's `scripts/inference/run_service.sh` + the
FastAPI service in `krisna_inference.orchestrator.service`).

`run_agentic_session.py` is a thin wrapper over the same
`SwapOrchestrator` class the real service uses — it does not
reimplement the agentic flow, so it can't silently drift from what
actually ships.

## Usage

```bash
# Safe default — MockBackend, no GPU, no models/ checkpoints needed.
# Verified working: exercises the full conversational-turn -> finalize ->
# critique sequence and prints each real result.
python inference-runtime/run_agentic_session.py --finalize --critique

# Real backends — GPU required, reads models/ checkpoints via the same
# KRISNA_* env vars scripts/inference/run_service.sh uses.
python inference-runtime/run_agentic_session.py --real --finalize

# Low-VRAM/CPU-offload mode (see docs/inference/) — verified: correctly
# applies the 12GB envelope and LOW_VRAM_REGISTRY figures.
python inference-runtime/run_agentic_session.py --real --low-vram --finalize

# Multi-turn scripted session, quality polish tier, with critique —
# useful as a smoke test after training a new checkpoint.
python inference-runtime/run_agentic_session.py --real \
    --message "a fintech dashboard" \
    --message "make the primary color teal" \
    --finalize --finalize-tier polish_quality --critique
```

## What it actually verifies vs. what it doesn't

**Verifies real orchestration behavior**: swap-tier exclusivity (baseline
unloads before Finalize/Critique load, per PRD §7.4), VRAM/RAM ledger
admission and rollback, fallback-tier degradation on OOM, the exact JSON
shape each `SwapResult` carries.

**Does NOT verify generation quality** — `--message`/`--finalize`/
`--critique` in MockBackend mode (the default) return scripted, fixed
mock output, not real model inference. Use `--real` with an actual GPU
and trained checkpoints to test real output; this harness's job in that
mode is giving you a fast, scriptable way to drive a full session without
standing up the FastAPI service and writing a client, not replacing real
evaluation.
