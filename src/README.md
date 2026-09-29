# src/ — superseded by inference-frontend/

This directory used to be a placeholder for a not-yet-built frontend
("scaffolded, design deferred"). That's no longer accurate: a real,
working frontend now exists at `../inference-frontend/` — a Node.js
control panel with a Setup/Install tab (drives
`scripts/inference/download_weights.py`) and a Studio tab (a full
session UI: chat, finalize, critique, live GPU/RAM utilization, a
pixel-forming render preview). See `../inference-frontend/README.md`.

This directory is kept only so old links/references to it don't 404 —
it has no code and isn't imported or run by anything. If you're looking
for the frontend, go to `../inference-frontend/` instead.

## For reference: this file's own stale claims, corrected

The version of this file that used to live here listed
`POST /conversation/turn`, `POST /finalize`, and `POST /critique` as the
service's real endpoints. Those never existed with those names — the
actual, current routes (verified directly against
`inference/src/krisna_inference/orchestrator/service.py`, not
re-described from memory) are session-scoped:

- `POST /session` — create a session
- `GET /session/{id}` — fetch its current state
- `GET /session/{id}/render` — the actual rendered image bytes, once finalized
- `POST /session/{id}/message` — a conversational turn
- `POST /session/{id}/finalize` — render
- `POST /session/{id}/critique` — critique a finalized render
- `GET /orchestrator/status` — VRAM/RAM, both the declared admission
  budget and live hardware probes
- `GET /health`, `GET/POST /preference-pairs/*`

See `tests/inference/test_service.py` for the exact real request/response
shapes, or just read `inference-frontend/public/app.js`, which already
drives every one of these from real, working frontend code.
