# Krisna Inference — Node.js control panel

A lightweight Node.js frontend (Express, one dependency) for the Python
inference service (`inference/src/krisna_inference/orchestrator/service.py`).
It does three jobs, and none of them touch MockBackend — this panel only
ever drives the real inference stack:

1. **Setup & Install tab** — drives `scripts/inference/download_weights.py`
   (spawned as a child process, streamed to the browser over Server-Sent
   Events) to download the four frozen HF-hosted models and locate this
   repo's own trained artifacts, then writes `.env.inference` at the repo
   root with `KRISNA_USE_REAL_BACKENDS=1`.
2. **Backend management** — starts/stops the real FastAPI/uvicorn service
   itself, as a child process, with `KRISNA_USE_REAL_BACKENDS=1` forced
   regardless of what's in `.env.inference`. If `.env.inference` already
   exists when `npm start` runs, the backend comes up automatically — no
   second terminal, no command to remember.
3. **Studio tab** — a thin proxy + UI over the FastAPI backend's session
   API (`/session`, `/message`, `/finalize`, `/critique`, `/orchestrator/status`).

This Node process holds no model weights and does no GPU work — it's a
control plane that launches and talks to the real Python inference layer.

## Prerequisites (once, on the machine that will actually run inference)

Real backends need a GPU and the full ML stack — this is not optional for
this panel, since it never falls back to mock:

```bash
# from the monorepo root
python3 -m venv .venv && source .venv/bin/activate
pip install -e inference/ -e training/
pip install -e "inference/[backends]"   # torch, transformers, diffusers, bitsandbytes...
```

(Critic runs in its own isolated venv — see
`scripts/training/setup_env_critic.sh` — because it needs a different
pinned `transformers` version than the other tiers. `download_weights.py`
checks for `./venv-critic/bin/python` and tells you if it's missing.)

## Run it

```bash
cd inference/frontend
npm install
npm start                 # http://127.0.0.1:3100
```

**First run** (nothing installed yet): the Setup tab is where you land.
Fill in options, optionally point at your trained Sketch checkpoint /
Polish LoRA (or leave blank — both auto-discover from the real training
output paths, see below), click **Start install**, watch it download.
The moment it finishes, this same process starts the real backend
automatically — the **Backend** panel's status pill flips to
"running (real backend)" with no further action.

**Every run after that**: `.env.inference` already exists, so `npm start`
brings the real backend up by itself, immediately. Open the Studio tab
and go.

## Studio Features

- **Canvas2D Multi-Pass Generalization Visualizer**: High-performance HTML5 Canvas rendering
  that visualizes generation dynamics — simulating progressive MaskGIT token unmasking
  during Sketch passes, dynamic flow-matching velocity particle trajectories during Polish,
  and rendering final high-fidelity pixels.
- **Active Constraints Bar**: Interactive chips dynamically displaying conversational intent,
  target device (`mobile_ios`, `desktop_web`), theme (`dark`, `light`), color palettes,
  and locked bounding box regions ($[x, y, w, h]$).
- **Closed Critic Iteration Loop**: Formatted critique display with severity badges and
  bounding box tags, equipped with a 1-click **"✨ Iterate with Critic Edits"** button
  that automatically feeds structured critique recommendations back into the Planner's
  conversational history for seamless multi-turn refinement (PRD §5.1–§5.3).
- **Hardware Telemetry & Ledger Status**: Real-time polling displaying declared VRAM/RAM
  admission ledger states alongside live `torch.cuda.mem_get_info()` and host `/proc/meminfo`
  hardware readings (PRD §7.2, §7.6).

Env vars (all optional):

| Var                       | Default                     | Meaning                                          |
|----------------------------|------------------------------|---------------------------------------------------|
| `PORT`                     | `3100`                       | This frontend's own port                           |
| `KRISNA_BACKEND_URL`       | `http://127.0.0.1:8420`      | Where the Studio tab's proxy sends session calls    |
| `KRISNA_BACKEND_HOST`      | `127.0.0.1`                  | Host this panel binds the backend to when it starts |
| `KRISNA_BACKEND_PORT`      | `8420`                       | Port this panel binds the backend to when it starts |
| `KRISNA_PYTHON_BIN`        | `<repo>/.venv/bin/python`    | Interpreter used for both the installer and the backend |
| `KRISNA_INFERENCE_APP_DIR` | (unset — normal package import) | Only set this if running against the source tree instead of a `pip install -e` |

## Where the real, this-repo-trained artifacts actually come from

Verified against the real training code and configs (not assumed) — see
`scripts/inference/download_weights.py`'s own comments for the full
citation trail:

| Artifact | Real on-disk location | Produced by |
|---|---|---|
| Sketch checkpoint | `checkpoints/sketch_stage2_512/checkpoint_final.pt` (or `sketch_stage1_256`) | `sketch/checkpoint_io.py`'s `save_checkpoint()`, called from `sketch/train.py` via `scripts/training/train_sketch_stage{1,2}.sh` |
| Polish LoRA (DPO-refined) | `models/dpo_checkpoints/<run>/final/` | `polish/train_dpo.py` via `scripts/training/train_polish_dpo.sh` |
| Polish LoRA (base fine-tune only) | `checkpoints/polish_default_lora/` | diffusers' `train_dreambooth_lora_z_image.py` via `scripts/training/train_polish_default_lora.sh` |

The installer auto-discovers both Polish LoRA shapes and prefers whichever
was modified most recently — normally the DPO-refined one, since DPO runs
on top of the base fine-tune.

## Why Node, and why this shape

- The install step is inherently a long-running process with incremental
  progress (multi-GB downloads, retries) — a plain HTML page hitting a
  Python `subprocess` and getting one blocking HTTP response back would
  either time out or show nothing until the end. Node's event loop makes
  "spawn a child process, stream its stdout as SSE to any number of
  connected browser tabs" a natural, small amount of code (see
  `server.js`'s `InstallJob` and `BackendProcess`).
- `scripts/inference/download_weights.py` emits one JSON object per stdout
  line specifically so this frontend (or any other consumer — CI, a
  human tailing logs) doesn't have to parse free-form text.
- The Studio tab intentionally proxies rather than reimplements: all real
  session logic (state machine, VRAM ledger, DesignState) stays in the
  Python orchestrator, which is the actual product of this repo's
  `inference/` package. This frontend never talks to a GPU, a model, or
  the SQLite session store directly — it manages the *process* that does.
- `KRISNA_USE_REAL_BACKENDS=1` is forced by `BackendProcess.start()`
  regardless of what's already in the environment or `.env.inference`.
  MockBackend remains a legitimate, useful mode inside
  `orchestrator/service.py` (tests rely on it), but this control panel
  has exactly one supported way to run the backend, and it isn't that
  one.

