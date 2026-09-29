# src/ — Frontend (scaffolded, design deferred)

This directory is intentionally close to empty. Per explicit instruction,
the frontend's design is deferred to a later pass — this is a placeholder
marking where it goes, not an implementation.

## What it will need to talk to

Whatever framework/approach gets chosen here, the actual integration
surface is already real and stable:

- **`krisna_inference.orchestrator.service`** — the FastAPI app
  (`scripts/inference/run_service.sh` runs it). Real, tested endpoints:
  `POST /conversation/turn`, `POST /finalize`, `POST /critique`,
  `GET /orchestrator/status` (now includes both the VRAM and RAM ledger
  snapshots — see `docs/inference/`). See `tests/inference/test_service.py`
  for the exact real request/response shapes each endpoint expects.
- **`inference-runtime/run_agentic_session.py`** — not something a
  frontend calls directly, but useful as a reference for the exact
  sequence a UI needs to drive: conversational turn(s) → optional
  finalize → optional critique, matching PRD §5.3's sequence flows.

## Suggested next step, when this gets designed

Point a dev server at the FastAPI service (`./scripts/inference/
run_service.sh`, MockBackend mode needs no GPU) and build against real
responses from day one, rather than mocking the API shape — the service
is already stable enough to develop against.
