// Lightweight Node.js frontend for the Krisna inference service.
//
// Two jobs, on purpose kept in one small Express app rather than a
// framework: (1) proxy the Studio UI's session calls to the real FastAPI
// backend (swap_orchestrator + design state), and (2) drive
// scripts/inference/download_weights.py as a child process so installing
// everything after training is a page, not a checklist of shell commands.
//
// This process holds NO model weights and does NO GPU work itself — it's
// a thin control-plane in front of the Python service, which is the
// actual inference layer (see inference/src/krisna_inference/orchestrator).

import express from "express";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const ENV_FILE = path.join(REPO_ROOT, ".env.inference");
const DOWNLOAD_SCRIPT = path.join(REPO_ROOT, "scripts", "inference", "download_weights.py");

const BACKEND_URL = process.env.KRISNA_BACKEND_URL || "http://127.0.0.1:8420";
const PORT = Number(process.env.PORT || 3100);
const PYTHON_BIN = process.env.KRISNA_PYTHON_BIN || path.join(REPO_ROOT, ".venv", "bin", "python");
const BACKEND_HOST = process.env.KRISNA_BACKEND_HOST || "127.0.0.1";
const BACKEND_PORT = process.env.KRISNA_BACKEND_PORT || "8420";
// Where the installed `krisna_inference`/`krisna_training` packages live.
// `pip install -e inference/` + `pip install -e training/` (see
// inference-frontend/README.md) puts them on the interpreter's own
// site-packages, so no --app-dir is needed by default; set this only if
// running uvicorn against the source tree directly instead of an install.
const BACKEND_APP_DIR = process.env.KRISNA_INFERENCE_APP_DIR || null;

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

function parseEnvFile() {
  if (!existsSync(ENV_FILE)) return null;
  const out = {};
  for (const line of readFileSync(ENV_FILE, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    out[trimmed.slice(0, eq)] = trimmed.slice(eq + 1);
  }
  return out;
}

// --------------------------------------------------------------------- //
// Backend process — the real FastAPI/uvicorn inference service. Managed
// here (not just documented as a second terminal command) so "npm start"
// is the whole seamless flow: install once in the Setup tab, then this
// process brings the real backend up itself, every time, with
// KRISNA_USE_REAL_BACKENDS=1 forced — MockBackend is a legitimate test
// mode inside the Python package (see orchestrator/service.py), but this
// control panel never launches it. If .env.inference doesn't exist yet
// (install hasn't run), this stays idle rather than guessing at env vars
// the installer is responsible for resolving.
// --------------------------------------------------------------------- //

class BackendProcess extends EventEmitter {
  constructor() {
    super();
    this.child = null;
    this.running = false;
    this.log = [];
    this.startedAt = null;
    this.exitCode = null;
    this.lastError = null;
  }

  start() {
    if (this.running) return { ok: false, error: "Backend is already running." };
    const envVars = parseEnvFile();
    if (!envVars) {
      return {
        ok: false,
        error: "No .env.inference found — run an install from the Setup tab first.",
      };
    }

    this.log = [];
    this.exitCode = null;
    this.lastError = null;
    this.startedAt = Date.now();

    const args = [
      "-m", "uvicorn",
      "krisna_inference.orchestrator.service:app",
      "--host", BACKEND_HOST,
      "--port", String(BACKEND_PORT),
    ];
    if (BACKEND_APP_DIR) args.push("--app-dir", BACKEND_APP_DIR);

    const childEnv = {
      ...process.env,
      ...envVars,
      // Forced, not just inherited from the env file: this control panel
      // has exactly one supported way to run the backend, and it isn't
      // the mock one. If envVars already says 1 (the installer always
      // writes it), this is a no-op; it only matters as a hard guarantee.
      KRISNA_USE_REAL_BACKENDS: "1",
    };

    const child = spawn(PYTHON_BIN, args, { cwd: REPO_ROOT, env: childEnv });
    this.running = true;
    this.child = child;
    this.emit("event", { event: "backend_starting", ts: Date.now() / 1000, host: BACKEND_HOST, port: BACKEND_PORT });

    const pipe = (stream) => {
      let buf = "";
      stream.on("data", (chunk) => {
        buf += chunk.toString("utf8");
        let idx;
        while ((idx = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          const evt = { event: "log", line, ts: Date.now() / 1000 };
          this.log.push(evt);
          this.emit("event", evt);
        }
      });
    };
    pipe(child.stdout);
    pipe(child.stderr);

    child.on("close", (code) => {
      this.running = false;
      this.exitCode = code;
      const evt = { event: "backend_exit", code, ts: Date.now() / 1000 };
      this.log.push(evt);
      this.emit("event", evt);
    });
    child.on("error", (err) => {
      this.running = false;
      this.lastError = String(err);
      const evt = { event: "backend_error", error: String(err), ts: Date.now() / 1000 };
      this.log.push(evt);
      this.emit("event", evt);
    });

    return { ok: true };
  }

  stop() {
    if (!this.running || !this.child) return { ok: false, error: "Backend is not running." };
    this.child.kill("SIGTERM");
    return { ok: true };
  }

  status() {
    return {
      running: this.running,
      startedAt: this.startedAt,
      exitCode: this.exitCode,
      lastError: this.lastError,
      host: BACKEND_HOST,
      port: BACKEND_PORT,
    };
  }
}

const backendProcess = new BackendProcess();

// Seamless direct running: if the install already happened in a previous
// session (.env.inference is on disk), bring the real backend up
// automatically on `npm start` rather than requiring a second terminal
// and a manually-remembered command. If it hasn't been installed yet,
// the Setup tab is exactly where a first-time operator should land
// anyway, so staying idle here is correct, not a bug.
if (parseEnvFile()) {
  const result = backendProcess.start();
  if (!result.ok) {
    console.warn(`Did not auto-start backend: ${result.error}`);
  } else {
    console.log(`Auto-starting real inference backend on ${BACKEND_HOST}:${BACKEND_PORT} (.env.inference found)`);
  }
} else {
  console.log("No .env.inference yet — open the Setup tab to install, then start the backend from there.");
}

app.get("/api/backend/status", (_req, res) => res.json(backendProcess.status()));

app.post("/api/backend/start", (_req, res) => {
  const result = backendProcess.start();
  res.status(result.ok ? 200 : 409).json(result);
});

app.post("/api/backend/stop", (_req, res) => {
  const result = backendProcess.stop();
  res.status(result.ok ? 200 : 409).json(result);
});

app.get("/api/backend/stream", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  for (const evt of backendProcess.log) res.write(`data: ${JSON.stringify(evt)}\n\n`);
  const onEvent = (evt) => res.write(`data: ${JSON.stringify(evt)}\n\n`);
  backendProcess.on("event", onEvent);
  req.on("close", () => backendProcess.off("event", onEvent));
});


// --------------------------------------------------------------------- //
// Install job — one at a time, in-memory. This is a single-operator,
// single-machine control panel (same "not a distributed system" spirit
// as the Python store.py), so one job slot is the right amount of
// complexity, not a queue.
// --------------------------------------------------------------------- //

class InstallJob extends EventEmitter {
  constructor() {
    super();
    this.running = false;
    this.log = []; // full history, replayed to new SSE clients on connect
    this.exitCode = null;
    this.startedAt = null;
    this.finishedAt = null;
  }

  start(args) {
    if (this.running) {
      throw new Error("An install is already running.");
    }
    if (!existsSync(DOWNLOAD_SCRIPT)) {
      throw new Error(`download_weights.py not found at ${DOWNLOAD_SCRIPT}`);
    }
    this.running = true;
    this.log = [];
    this.exitCode = null;
    this.startedAt = Date.now();
    this.finishedAt = null;

    const child = spawn(PYTHON_BIN, [DOWNLOAD_SCRIPT, ...args], {
      cwd: REPO_ROOT,
      env: process.env,
    });

    let stdoutBuf = "";
    child.stdout.on("data", (chunk) => {
      stdoutBuf += chunk.toString("utf8");
      let idx;
      while ((idx = stdoutBuf.indexOf("\n")) !== -1) {
        const line = stdoutBuf.slice(0, idx).trim();
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (!line) continue;
        this._handleLine(line);
      }
    });

    let stderrBuf = "";
    child.stderr.on("data", (chunk) => {
      stderrBuf += chunk.toString("utf8");
      let idx;
      while ((idx = stderrBuf.indexOf("\n")) !== -1) {
        const line = stderrBuf.slice(0, idx).trim();
        stderrBuf = stderrBuf.slice(idx + 1);
        if (!line) continue;
        // stderr lines aren't the JSON-event protocol (they're tracebacks,
        // pip noise, etc.) — surface them as a distinct event type rather
        // than trying to force-parse them as JSON.
        this._handleEvent({ event: "stderr", line, ts: Date.now() / 1000 });
      }
    });

    child.on("close", (code) => {
      this.running = false;
      this.exitCode = code;
      this.finishedAt = Date.now();
      this._handleEvent({ event: "process_exit", code, ts: Date.now() / 1000 });
    });

    child.on("error", (err) => {
      this.running = false;
      this.exitCode = -1;
      this.finishedAt = Date.now();
      this._handleEvent({ event: "process_error", error: String(err), ts: Date.now() / 1000 });
    });

    this.child = child;
  }

  _handleLine(line) {
    try {
      this._handleEvent(JSON.parse(line));
    } catch {
      // download_weights.py's contract is one JSON object per stdout line;
      // anything that fails to parse is surfaced as-is rather than dropped
      // silently, so a bug in that contract is visible instead of hidden.
      this._handleEvent({ event: "raw", line, ts: Date.now() / 1000 });
    }
  }

  _handleEvent(evt) {
    this.log.push(evt);
    this.emit("event", evt);
  }

  status() {
    return {
      running: this.running,
      exitCode: this.exitCode,
      startedAt: this.startedAt,
      finishedAt: this.finishedAt,
      eventCount: this.log.length,
    };
  }
}

const installJob = new InstallJob();

// Seamless hand-off: the moment an install finishes successfully, bring
// the real backend up immediately — no separate click, no restart of
// this frontend required. (A prior *partial* install_incomplete never
// triggers this — see write_env_file()'s own note on why a partial
// .env.inference is still written: we still don't want to silently
// launch the backend against a config known to be missing something.)
installJob.on("event", (evt) => {
  if (evt.event === "install_complete" && !backendProcess.running) {
    const result = backendProcess.start();
    if (result.ok) console.log("Install complete — starting real inference backend.");
  }
});

// --------------------------------------------------------------------- //
// Install API
// --------------------------------------------------------------------- //

app.get("/api/install/status", (_req, res) => {
  res.json({
    job: installJob.status(),
    envFile: parseEnvFile(),
    envFilePath: ENV_FILE,
  });
});

app.post("/api/install/start", (req, res) => {
  const {
    lowVram = false,
    fastPlanner = false,
    skipQuality = false,
    skipCritic = false,
    sketchCheckpoint = null,
    polishLora = null,
    dryRun = false,
  } = req.body || {};

  const args = [];
  if (lowVram) args.push("--low-vram");
  if (fastPlanner) args.push("--fast-planner");
  if (skipQuality) args.push("--skip-quality");
  if (skipCritic) args.push("--skip-critic");
  if (sketchCheckpoint) args.push("--sketch-checkpoint", sketchCheckpoint);
  if (polishLora) args.push("--polish-lora", polishLora);
  if (dryRun) args.push("--dry-run");

  try {
    installJob.start(args);
    res.json({ ok: true, args });
  } catch (e) {
    res.status(409).json({ ok: false, error: String(e.message || e) });
  }
});

// Server-Sent Events: replay history, then stream new events live. Chosen
// over WebSockets because this is one-directional (server -> browser) log
// tailing — SSE is the simpler primitive for exactly that, with automatic
// browser reconnect built in, and it needs zero extra npm dependencies.
app.get("/api/install/stream", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  for (const evt of installJob.log) {
    res.write(`data: ${JSON.stringify(evt)}\n\n`);
  }

  const onEvent = (evt) => res.write(`data: ${JSON.stringify(evt)}\n\n`);
  installJob.on("event", onEvent);

  req.on("close", () => {
    installJob.off("event", onEvent);
  });
});

// --------------------------------------------------------------------- //
// Studio API — thin proxy to the FastAPI backend (swap_orchestrator +
// design_state). Kept as explicit routes (not a blanket proxy-everything)
// so this frontend has one obvious place per backend capability, and
// never accidentally exposes an endpoint the backend didn't intend for
// browser callers.
// --------------------------------------------------------------------- //

async function proxyJson(res, method, url, body) {
  try {
    const r = await fetch(`${BACKEND_URL}${url}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await r.text();
    res.status(r.status);
    res.type("application/json");
    res.send(text || "{}");
  } catch (e) {
    res.status(502).json({
      error: "backend_unreachable",
      detail: String(e.message || e),
      backend_url: BACKEND_URL,
      hint: "Is the Python service running? ./scripts/inference/run_service.sh",
    });
  }
}

app.get("/api/health", (_req, res) => proxyJson(res, "GET", "/health"));
app.get("/api/status", (_req, res) => proxyJson(res, "GET", "/orchestrator/status"));

app.post("/api/session", (req, res) => proxyJson(res, "POST", "/session", req.body));
app.get("/api/session/:id", (req, res) => proxyJson(res, "GET", `/session/${req.params.id}`));
app.post("/api/session/:id/message", (req, res) =>
  proxyJson(res, "POST", `/session/${req.params.id}/message`, req.body));
app.post("/api/session/:id/finalize", (req, res) =>
  proxyJson(res, "POST", `/session/${req.params.id}/finalize`, req.body));
app.post("/api/session/:id/critique", (req, res) =>
  proxyJson(res, "POST", `/session/${req.params.id}/critique`, req.body || {}));

app.get("/api/preference-pairs/stats", (_req, res) => proxyJson(res, "GET", "/preference-pairs/stats"));

app.listen(PORT, () => {
  console.log(`Krisna inference frontend listening on http://127.0.0.1:${PORT}`);
  console.log(`Proxying Studio API calls to backend at ${BACKEND_URL}`);
  console.log(`Install job runs: ${PYTHON_BIN} ${DOWNLOAD_SCRIPT}`);
});

// One command, one thing to stop: Ctrl+C here also stops the real
// backend this process launched, rather than leaving a uvicorn process
// orphaned and holding the GPU/port after the frontend exits.
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    if (backendProcess.running) {
      console.log("Stopping backend…");
      backendProcess.stop();
    }
    process.exit(0);
  });
}

