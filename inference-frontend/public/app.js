// Krisna inference control panel — frontend logic.
//
// Three visual systems live here, each mirroring a real backend concept
// rather than decorating one:
//   1. The install tracker  — mirrors download_weights.py's actual step
//      sequence (see INSTALL_STEPS below), driven by its JSON event log.
//   2. The GPU memory rack  — mirrors model_registry.py's tier sizes and
//      swap_orchestrator.py's ledger snapshot (which tiers are resident).
//   3. The residency diagram — mirrors state_machine.py's TRANSITIONS
//      table exactly (same states, same edges).

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ------------------------------------------------------------------ //
// Tabs
// ------------------------------------------------------------------ //
$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.remove("active"));
    $$(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "studio") refreshStatus();
  });
});

// ------------------------------------------------------------------ //
// Health pill
// ------------------------------------------------------------------ //
async function refreshHealth() {
  const pill = $("#health-pill");
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    pill.innerHTML = `<i class="dot"></i>${data.residency_state}`;
    pill.className = "pill ok";
  } catch (e) {
    pill.innerHTML = `<i class="dot"></i>backend unreachable`;
    pill.className = "pill err";
  }
}
refreshHealth();
setInterval(refreshHealth, 5000);

// ==================================================================== //
// 1. Install tracker
// ==================================================================== //

// The real, ordered sequence download_weights.py runs. `match` decides
// which JSON events belong to this step; `tier`/`label` narrow which
// event instance (multiple models share the "model_download_*" events).
const INSTALL_STEPS = [
  { id: "plan", title: "Plan & disk check", match: (e) => e.event === "plan" || e.event === "disk_check" },
  { id: "planner", title: "Download Planner", tier: "planner", match: (e) => e.tier === "planner" },
  { id: "polish_default", title: "Download Polish · Default", tier: "polish_default", match: (e) => e.tier === "polish_default" },
  { id: "polish_quality", title: "Download Polish · Quality", tier: "polish_quality", match: (e) => e.tier === "polish_quality", skippable: "skipQuality" },
  { id: "critic", title: "Download Critic", tier: "critic", match: (e) => e.tier === "critic", skippable: "skipCritic" },
  { id: "sketch_ckpt", title: "Locate Sketch checkpoint", match: (e) => e.label === "SKETCH_CHECKPOINT" },
  { id: "polish_lora", title: "Locate Polish LoRA", match: (e) => e.label === "POLISH_LORA" },
  { id: "critic_venv", title: "Verify Critic venv", match: (e) => e.event === "critic_venv_found" || e.event === "critic_venv_missing", skippable: "skipCritic" },
  { id: "env_write", title: "Write .env.inference", match: (e) => e.event === "env_file_written" || e.event === "install_incomplete" || e.event === "install_complete" || e.event === "dry_run_complete" },
];

function newStepState() {
  return Object.fromEntries(INSTALL_STEPS.map((s) => [s.id, { status: "pending", detail: "" }]));
}
let stepState = newStepState();
let lastFormFlags = {};

function applyEventToSteps(evt) {
  for (const step of INSTALL_STEPS) {
    if (!step.match(evt)) continue;
    const s = stepState[step.id];
    switch (evt.event) {
      case "plan":
        s.status = "active"; s.detail = `envelope target reachable for: ${evt.models?.join(", ")}`;
        break;
      case "disk_check":
        s.status = evt.ok ? "done" : "failed";
        s.detail = `${evt.free_gb}GB free / ${evt.needed_gb}GB needed`;
        break;
      case "model_download_start":
        s.status = "active"; s.detail = `${evt.repo_id} downloading…`;
        break;
      case "model_download_retry":
        s.status = "active"; s.detail = `retry ${evt.attempt}/${evt.max_retries}: ${evt.error}`;
        break;
      case "model_download_complete":
        s.status = "done"; s.detail = evt.path;
        break;
      case "model_download_failed":
        s.status = "failed"; s.detail = evt.error;
        break;
      case "model_download_skipped_dry_run":
        s.status = "done"; s.detail = "would download (dry run)";
        break;
      case "local_artifact_found":
        s.status = "done"; s.detail = `${evt.path} (${evt.source})`;
        break;
      case "local_artifact_missing":
        s.status = evt.required ? "failed" : "skipped";
        s.detail = evt.required ? "not found — required" : "not found — optional, continuing";
        break;
      case "critic_venv_found":
        s.status = "done"; s.detail = evt.path;
        break;
      case "critic_venv_missing":
        s.status = "failed"; s.detail = evt.hint;
        break;
      case "env_file_written":
        s.status = "done"; s.detail = `${evt.vars} vars written`;
        break;
      case "install_incomplete":
        if (s.status !== "done") s.status = "failed";
        break;
      case "dry_run_complete":
        s.status = evt.would_fail?.length ? "failed" : "done";
        break;
    }
  }
}

function applySkipFlags(flags) {
  for (const step of INSTALL_STEPS) {
    if (step.skippable && flags[step.skippable]) {
      stepState[step.id].status = "skipped";
      stepState[step.id].detail = "skipped by option";
    }
  }
}

function renderTracker() {
  const ol = $("#tracker");
  ol.innerHTML = "";
  INSTALL_STEPS.forEach((step, i) => {
    const s = stepState[step.id];
    const li = document.createElement("li");
    li.className = `tracker-step ${s.status}`;
    li.innerHTML = `
      <span class="step-dot"></span>
      <div class="step-title">${i + 1}. ${step.title}</div>
      ${s.detail ? `<div class="step-detail">${escapeHtml(s.detail)}</div>` : ""}
    `;
    ol.appendChild(li);
  });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

renderTracker();

function appendLog(line) {
  const pre = $("#install-log");
  pre.textContent += line + "\n";
  pre.scrollTop = pre.scrollHeight;
}

function describeEvent(evt) {
  switch (evt.event) {
    case "plan": return `Planning install — models: ${evt.models?.join(", ")}`;
    case "disk_check": return `Disk check: need ~${evt.needed_gb}GB, have ${evt.free_gb}GB free — ${evt.ok ? "OK" : "INSUFFICIENT"}`;
    case "model_download_start": return `Downloading ${evt.repo_id} (${evt.tier})…`;
    case "model_download_retry": return `Retry ${evt.attempt}/${evt.max_retries} for ${evt.repo_id}: ${evt.error}`;
    case "model_download_complete": return `Done: ${evt.repo_id} → ${evt.path}`;
    case "model_download_failed": return `FAILED: ${evt.repo_id} — ${evt.error}`;
    case "model_download_skipped_dry_run": return `(dry run) would download ${evt.repo_id}`;
    case "local_artifact_found": return `Found ${evt.label} at ${evt.path} (${evt.source})`;
    case "local_artifact_missing": return `${evt.required ? "MISSING (required)" : "Not found (optional)"}: ${evt.label}`;
    case "critic_venv_found": return `Critic venv OK: ${evt.path}`;
    case "critic_venv_missing": return `Critic venv missing at ${evt.path} — ${evt.hint}`;
    case "env_file_written": return `Wrote ${evt.path} (${evt.vars} vars)`;
    case "install_incomplete": return `Install incomplete — ${evt.failures.length} issue(s):\n  ${evt.failures.join("\n  ")}`;
    case "install_complete": return `Install complete → ${evt.env_file}`;
    case "dry_run_complete": return evt.would_fail.length ? `Dry run found issues:\n  ${evt.would_fail.join("\n  ")}` : "Dry run OK — everything resolvable.";
    case "process_exit": return `--- process exited with code ${evt.code} ---`;
    case "process_error": return `--- process error: ${evt.error} ---`;
    case "stderr": return `[stderr] ${evt.line}`;
    case "raw": return evt.line;
    default: return JSON.stringify(evt);
  }
}

let installSource = null;

function connectInstallStream() {
  if (installSource) installSource.close();
  installSource = new EventSource("/api/install/stream");
  installSource.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    applyEventToSteps(evt);
    renderTracker();
    appendLog(describeEvent(evt));
    if (["install_complete", "install_incomplete", "dry_run_complete"].includes(evt.event)) {
      $("#log-status").textContent = evt.event === "install_complete" ? "complete" : "needs attention";
      refreshEnvTable();
    }
  };
}

function formValues() {
  const fd = new FormData($("#install-form"));
  return {
    lowVram: fd.has("lowVram"),
    fastPlanner: fd.has("fastPlanner"),
    skipQuality: fd.has("skipQuality"),
    skipCritic: fd.has("skipCritic"),
    sketchCheckpoint: fd.get("sketchCheckpoint") || null,
    polishLora: fd.get("polishLora") || null,
  };
}

async function startInstall(dryRun) {
  stepState = newStepState();
  const flags = formValues();
  lastFormFlags = flags;
  applySkipFlags(flags);
  renderTracker();
  $("#install-log").textContent = "";
  $("#log-status").textContent = dryRun ? "dry run…" : "running…";

  const r = await fetch("/api/install/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...flags, dryRun }),
  });
  const data = await r.json();
  if (!r.ok) {
    appendLog(`Could not start: ${data.error}`);
    $("#log-status").textContent = "error";
    return;
  }
  connectInstallStream();
}

$("#install-form").addEventListener("submit", (e) => { e.preventDefault(); startInstall(false); });
$("#btn-dry-run").addEventListener("click", () => startInstall(true));

async function refreshEnvTable() {
  const r = await fetch("/api/install/status");
  const data = await r.json();
  $("#env-file-path").textContent = data.envFilePath;
  const tbody = $("#env-table tbody");
  tbody.innerHTML = "";
  if (!data.envFile) {
    tbody.innerHTML = `<tr><td colspan="2" class="hint">Not written yet — run an install.</td></tr>`;
    return;
  }
  for (const [k, v] of Object.entries(data.envFile)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${escapeHtml(v)}</td>`;
    tbody.appendChild(tr);
  }
}
$("#btn-refresh-env").addEventListener("click", refreshEnvTable);
refreshEnvTable();

// ------------------------------------------------------------------ //
// Backend panel — the real inference service, managed by server.js.
// ------------------------------------------------------------------ //

let backendSource = null;

function appendBackendLog(text) {
  const pre = $("#backend-log");
  pre.textContent += text + "\n";
  pre.scrollTop = pre.scrollHeight;
}

function describeBackendEvent(evt) {
  switch (evt.event) {
    case "backend_starting": return `Starting uvicorn on ${evt.host}:${evt.port}…`;
    case "log": return evt.line;
    case "backend_exit": return `--- backend exited with code ${evt.code} ---`;
    case "backend_error": return `--- backend error: ${evt.error} ---`;
    default: return JSON.stringify(evt);
  }
}

function connectBackendStream() {
  if (backendSource) backendSource.close();
  backendSource = new EventSource("/api/backend/stream");
  backendSource.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    appendBackendLog(describeBackendEvent(evt));
    refreshBackendStatus();
  };
}

async function refreshBackendStatus() {
  const r = await fetch("/api/backend/status");
  const data = await r.json();
  const pill = $("#backend-pill");
  if (data.running) {
    pill.innerHTML = `<i class="dot"></i>running (real backend)`;
    pill.className = "pill ok";
  } else if (data.lastError) {
    pill.innerHTML = `<i class="dot"></i>error`;
    pill.className = "pill err";
  } else if (data.exitCode !== null && data.exitCode !== 0) {
    pill.innerHTML = `<i class="dot"></i>exited (${data.exitCode})`;
    pill.className = "pill err";
  } else {
    pill.innerHTML = `<i class="dot"></i>stopped`;
    pill.className = "pill unknown";
  }
}

$("#btn-backend-start").addEventListener("click", async () => {
  $("#backend-log").textContent = "";
  const r = await fetch("/api/backend/start", { method: "POST" });
  const data = await r.json();
  if (!r.ok) appendBackendLog(`Could not start: ${data.error}`);
  connectBackendStream();
  refreshBackendStatus();
});

$("#btn-backend-stop").addEventListener("click", async () => {
  await fetch("/api/backend/stop", { method: "POST" });
  refreshBackendStatus();
});

connectBackendStream();
refreshBackendStatus();
setInterval(refreshBackendStatus, 4000);

// ==================================================================== //
// 2. GPU memory rack — mirrors model_registry.py's REGISTRY /
//    LOW_VRAM_REGISTRY declared sizes, combined with live residency
//    from /orchestrator/status. Client-side numbers are a presentation
//    mirror only — the backend's ledger is always the source of truth
//    for admission decisions.
// ==================================================================== //

const TIER_REGISTRY = {
  // Kept in sync by hand with orchestrator/model_registry.py's REGISTRY /
  // LOW_VRAM_REGISTRY — this mirror has drifted stale at least once
  // before (polish_default's vram_gb sat at a pre-Phase-13 8.0 here
  // after the backend was corrected to 14.0; critic's low-vram figures
  // also drifted back to a since-fixed 12.0/40.0 pair on the backend
  // side independently of this file). No shared source of truth between
  // this static object and the Python registry currently exists — if
  // GET /orchestrator/status ever starts including the resolved
  // per-tier vram_gb/ram_gb figures directly, this object should be
  // deleted in favor of that, rather than hand-copied again.
  full: {
    envelope_gb: 24.0,
    tiers: [
      { key: "planner", label: "Planner", vram_gb: 6.5, ram_gb: 0.0, always_resident: true, color: "#f2a154" },
      { key: "sketch", label: "Sketch", vram_gb: 3.0, ram_gb: 0.0, always_resident: true, color: "#e8935f" },
      { key: "polish_default", label: "Polish · Default", vram_gb: 14.0, ram_gb: 0.0, always_resident: false, color: "#55c2c0" },
      { key: "polish_quality", label: "Polish · Quality", vram_gb: 16.0, ram_gb: 0.0, always_resident: false, color: "#4aa9c9" },
      { key: "critic", label: "Critic", vram_gb: 18.0, ram_gb: 0.0, always_resident: false, color: "#8f7fd6" },
    ],
  },
  low_vram: {
    envelope_gb: 12.0,
    ram_envelope_gb: 48.0,
    tiers: [
      { key: "planner", label: "Planner", vram_gb: 6.5, ram_gb: 0.0, always_resident: true, color: "#f2a154" },
      { key: "sketch", label: "Sketch", vram_gb: 3.0, ram_gb: 0.0, always_resident: true, color: "#e8935f" },
      { key: "polish_default", label: "Polish · Default", vram_gb: 12.0, ram_gb: 2.0, always_resident: false, color: "#55c2c0" },
      { key: "polish_quality", label: "Polish · Quality", vram_gb: 10.0, ram_gb: 10.0, always_resident: false, color: "#4aa9c9" },
      { key: "critic", label: "Critic", vram_gb: 11.5, ram_gb: 45.0, always_resident: false, color: "#8f7fd6" },
    ],
  },
};

function renderRack(status) {
  const mode = status?.low_vram_mode ? "low_vram" : "full";
  const reg = TIER_REGISTRY[mode];
  const envelopeGb = status?.vram?.envelope_gb ?? reg.envelope_gb;
  const usedGb = status?.vram?.used_gb ?? 0;
  const resident = new Set(status?.vram?.resident ?? []);
  const inFlight = status?.residency_state && status.residency_state.startsWith("swapping");

  $("#rack-envelope-label").innerHTML =
    `<span>VRAM envelope (declared budget)</span><strong>${usedGb.toFixed(1)} / ${envelopeGb.toFixed(1)} GB</strong>`;

  const rack = $("#rack");
  rack.innerHTML = "";
  for (const tier of reg.tiers) {
    const seg = document.createElement("div");
    const isResident = resident.has(tier.key);
    const isLoading = inFlight && !isResident && (tier.key === "polish_default" || tier.key === "polish_quality" || tier.key === "critic");
    let stateClass = "state-idle";
    if (isResident) stateClass = "state-resident";
    else if (isLoading) stateClass = "state-loading";

    seg.className = `rack-seg ${stateClass}`;
    seg.style.flexBasis = `${(tier.vram_gb / envelopeGb) * 100}%`;
    if (isResident) seg.style.background = tier.color;
    seg.innerHTML = `<span class="seg-label">${tier.label} · ${tier.vram_gb}GB</span>`;
    rack.appendChild(seg);
  }

  const legend = $("#rack-legend");
  legend.innerHTML = "";
  for (const tier of reg.tiers) {
    const li = document.createElement("li");
    const isResident = resident.has(tier.key);
    li.innerHTML = `<span class="legend-swatch" style="background:${isResident ? tier.color : "transparent"};border:1px solid ${tier.color}"></span>${tier.label}${tier.always_resident ? " · baseline" : ""}`;
    legend.appendChild(li);
  }

  renderHardwareBar("vram", status?.real_vram, usedGb);
}

// RAM-offload rack — same segmented-bar visual language as the VRAM
// rack above, but for orchestrator.ram_ledger (system RAM used by
// CPU-offloaded tier weights). In full-VRAM mode every tier's ram_gb is
// 0.0 by design (see model_registry.py) — shown as an empty, honestly
// labeled rack rather than hidden, so "offload isn't happening" is a
// visible state, not an absent one.
function renderRamRack(status) {
  const mode = status?.low_vram_mode ? "low_vram" : "full";
  const reg = TIER_REGISTRY[mode];
  const ramEnvelopeGb = status?.ram?.envelope_gb ?? reg.ram_envelope_gb ?? 48.0;
  const ramUsedGb = status?.ram?.used_gb ?? 0;
  const resident = new Set(status?.ram?.resident ?? []);
  const offloadTiers = reg.tiers.filter((t) => t.ram_gb > 0);

  $("#ram-rack-label").innerHTML =
    `<span>RAM offload (declared budget)</span><strong>${ramUsedGb.toFixed(1)} / ${ramEnvelopeGb.toFixed(1)} GB</strong>`;

  const rack = $("#ram-rack");
  rack.innerHTML = "";
  if (offloadTiers.length === 0) {
    rack.innerHTML = `<div class="rack-seg state-idle" style="flex-basis:100%"><span class="seg-label">no tier offloads to RAM in this mode</span></div>`;
  } else {
    for (const tier of offloadTiers) {
      const seg = document.createElement("div");
      const isResident = resident.has(tier.key);
      seg.className = `rack-seg ${isResident ? "state-resident" : "state-idle"}`;
      seg.style.flexBasis = `${(tier.ram_gb / ramEnvelopeGb) * 100}%`;
      if (isResident) seg.style.background = tier.color;
      seg.innerHTML = `<span class="seg-label">${tier.label} · ${tier.ram_gb}GB</span>`;
      rack.appendChild(seg);
    }
  }

  renderHardwareBar("ram", status?.real_ram, ramUsedGb);
}

// Live hardware readout — the actual torch.cuda.mem_get_info()/
// /proc/meminfo numbers from probe_real_vram()/probe_real_ram(), shown
// alongside (never instead of) the declared-budget racks above. These
// two numbers can legitimately disagree — the ledger is an admission
// bookkeeping abstraction (declared vram_gb per tier), the hardware
// probe is what's actually free on the card/host right now, including
// anything outside this process entirely (another process on a shared
// GPU, OS memory pressure, CUDA context overhead the ledger's estimates
// don't capture per Phase 6's own disclosed limitation).
function renderHardwareBar(kind, probe, ledgerUsedGb) {
  const fill = $(`#hw-${kind}-fill`);
  const reading = $(`#hw-${kind}-reading`);
  const block = $(`#hw-${kind}-block`);
  if (!probe) {
    block.classList.add("unavailable");
    fill.style.width = "0%";
    reading.textContent = "not available on this host";
    return;
  }
  block.classList.remove("unavailable");
  const usedGb = Math.max(0, probe.total_gb - probe.free_gb);
  const pct = probe.total_gb > 0 ? Math.min(100, Math.round((usedGb / probe.total_gb) * 100)) : 0;
  fill.style.width = `${pct}%`;
  reading.textContent = `${usedGb.toFixed(1)} / ${probe.total_gb.toFixed(1)} GB (${pct}%)`;
}

// ==================================================================== //
// 3. Residency state diagram — same states/edges as
//    orchestrator/state_machine.py's TRANSITIONS table.
// ==================================================================== //

const SD_NODES = {
  IDLE_RESIDENT: { x: 190, y: 10, w: 130, label: "IDLE RESIDENT" },
  SWAPPING_TO_POLISH: { x: 10, y: 80, w: 130, label: "SWAPPING → POLISH" },
  POLISH_RESIDENT: { x: 10, y: 150, w: 130, label: "POLISH RESIDENT" },
  SWAPPING_BACK_FROM_POLISH: { x: 10, y: 220, w: 130, label: "SWAP BACK ← POLISH" },
  SWAPPING_TO_CRITIC: { x: 370, y: 80, w: 130, label: "SWAPPING → CRITIC" },
  CRITIC_RESIDENT: { x: 370, y: 150, w: 130, label: "CRITIC RESIDENT" },
  SWAPPING_BACK_FROM_CRITIC: { x: 370, y: 220, w: 130, label: "SWAP BACK ← CRITIC" },
  ERROR_RECOVERY: { x: 190, y: 290, w: 130, label: "ERROR RECOVERY" },
};

const SD_EDGES = [
  ["IDLE_RESIDENT", "SWAPPING_TO_POLISH"],
  ["SWAPPING_TO_POLISH", "POLISH_RESIDENT"],
  ["POLISH_RESIDENT", "SWAPPING_BACK_FROM_POLISH"],
  ["SWAPPING_BACK_FROM_POLISH", "IDLE_RESIDENT"],
  ["IDLE_RESIDENT", "SWAPPING_TO_CRITIC"],
  ["SWAPPING_TO_CRITIC", "CRITIC_RESIDENT"],
  ["CRITIC_RESIDENT", "SWAPPING_BACK_FROM_CRITIC"],
  ["SWAPPING_BACK_FROM_CRITIC", "IDLE_RESIDENT"],
  ["SWAPPING_TO_POLISH", "ERROR_RECOVERY"],
  ["SWAPPING_TO_CRITIC", "ERROR_RECOVERY"],
  ["ERROR_RECOVERY", "IDLE_RESIDENT"],
];

function nodeCenter(n) { return { x: n.x + n.w / 2, y: n.y + 16 }; }

function renderStateDiagram(currentStateRaw) {
  const current = (currentStateRaw || "idle_resident").toUpperCase();
  const h = 340;
  let svg = `<svg viewBox="0 0 520 ${h}" xmlns="http://www.w3.org/2000/svg">`;

  for (const [fromKey, toKey] of SD_EDGES) {
    const a = nodeCenter(SD_NODES[fromKey]);
    const b = nodeCenter(SD_NODES[toKey]);
    const active = fromKey === current || toKey === current;
    svg += `<line class="sd-edge${active ? " active" : ""}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" marker-end="url(#arrow)" />`;
  }

  svg += `<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#232a35" /></marker></defs>`;

  for (const [key, n] of Object.entries(SD_NODES)) {
    const isCurrent = key === current;
    svg += `<g class="sd-node${isCurrent ? " current" : ""}">
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="32" rx="4" />
      <text x="${n.x + n.w / 2}" y="${n.y + 20}" text-anchor="middle">${n.label}</text>
    </g>`;
  }

  svg += `</svg>`;
  $("#state-diagram").innerHTML = svg;
}

// ------------------------------------------------------------------ //
// Studio: orchestrator status polling
// ------------------------------------------------------------------ //

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const data = await r.json();
    renderRack(data);
    renderRamRack(data);
    renderStateDiagram(data.residency_state);
  } catch (e) {
    // backend unreachable — leave last-known visuals in place, health
    // pill already communicates the outage.
  }
}
$("#btn-refresh-status").addEventListener("click", refreshStatus);
setInterval(() => { if ($("#tab-studio").classList.contains("active")) refreshStatus(); }, 4000);
renderRack(null);
renderRamRack(null);
renderStateDiagram("idle_resident");
renderPhasePipeline(null);
drawPlaceholderGrid(null, "idle");

// ------------------------------------------------------------------ //
// Studio: session lifecycle
// ------------------------------------------------------------------ //

const STAGES = ["conversing", "sketching", "finalizing", "finalized", "critiquing"];
let sessionState = null;

// ==================================================================== //
// Generation pipeline — phase nodes (Planner -> Sketch -> Polish ->
// Critic), mirroring DesignState.stage transitions in
// orchestrator/flows.py, not an invented decoration. "finalizing" maps
// to Polish being active; "finalized" means Polish has already produced
// a result (shown as its done state, ready for Critique next).
// ==================================================================== //

const PHASE_NODES = [
  { key: "planner", label: "PLANNER", sub: "Qwen3.5-9B", x: 10 },
  { key: "sketch", label: "SKETCH", sub: "VQ tokens", x: 145 },
  { key: "polish", label: "POLISH", sub: "Z-Image-Turbo", x: 280 },
  { key: "critic", label: "CRITIC", sub: "Gemma 4", x: 415 },
];
const NODE_W = 110, NODE_H = 40;

function phaseStatusFor(stage) {
  // Returns {planner, sketch, polish, critic}, each "pending"|"active"|"done".
  if (!stage) return { planner: "pending", sketch: "pending", polish: "pending", critic: "pending" };
  const order = ["conversing", "sketching", "finalizing", "finalized", "critiquing"];
  const idx = order.indexOf(stage);
  const s = { planner: "done", sketch: "pending", polish: "pending", critic: "pending" };
  if (idx <= 0) { s.planner = "active"; return s; }
  s.sketch = idx === 1 ? "active" : "done";
  if (idx === 2) { s.polish = "active"; return s; }
  s.polish = idx >= 3 ? "done" : "pending";
  s.critic = idx === 4 ? "active" : (idx > 4 ? "done" : "pending");
  return s;
}

function renderPhasePipeline(stage) {
  const status = phaseStatusFor(stage);
  $("#pipeline-phase-label").textContent = stage || "idle";
  let svg = `<svg viewBox="0 0 545 60" xmlns="http://www.w3.org/2000/svg">`;
  for (let i = 0; i < PHASE_NODES.length - 1; i++) {
    const a = PHASE_NODES[i], b = PHASE_NODES[i + 1];
    const active = status[a.key] === "done" || (status[a.key] === "active" && status[b.key] !== "pending");
    svg += `<line class="phase-edge${active ? " active" : ""}" x1="${a.x + NODE_W}" y1="30" x2="${b.x}" y2="30" />`;
  }
  for (const node of PHASE_NODES) {
    const st = status[node.key];
    svg += `<g class="phase-node ${st}">
      <rect x="${node.x}" y="10" width="${NODE_W}" height="${NODE_H}" rx="5" />
      <circle class="phase-dot" cx="${node.x + 10}" cy="20" r="3" />
      <text class="phase-title" x="${node.x + NODE_W / 2}" y="26" text-anchor="middle">${node.label}</text>
      <text class="phase-sub" x="${node.x + NODE_W / 2}" y="40" text-anchor="middle">${node.sub}</text>
    </g>`;
  }
  svg += `</svg>`;
  $("#phase-pipeline").innerHTML = svg;
}

// ==================================================================== //
// Pixel-forming canvas. Two honest states, never a fabricated one:
//   - No render yet / mid-finalize: a deterministic placeholder pixel
//     grid (derived from the session id, so it's stable per session
//     rather than random noise every redraw), pixelated + blurred to
//     read as "not formed yet."
//   - A real render exists: fetches the ACTUAL image bytes from
//     GET /api/session/:id/render (added alongside this feature — see
//     service.py's new /session/{id}/render route) and reveals it with
//     a blur-to-sharp transition. Never shows a fake "generated" image.
// ==================================================================== //

function seededRand(seed) {
  let x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function drawPlaceholderGrid(sessionId, tone) {
  const canvas = $("#pixel-canvas");
  const ctx = canvas.getContext("2d");
  const grid = 12;
  const cell = canvas.width / grid;
  const seedBase = [...(sessionId || "seed")].reduce((a, c) => a + c.charCodeAt(0), 0);
  const hue = tone === "forming" ? 28 : 210; // amber while forming, cool slate at rest
  for (let y = 0; y < grid; y++) {
    for (let x = 0; x < grid; x++) {
      const r = seededRand(seedBase + x * 13.1 + y * 7.7);
      const light = 12 + r * 14;
      ctx.fillStyle = `hsl(${hue}, ${tone === "forming" ? 45 : 15}%, ${light}%)`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}

let lastRenderedImageRef = null;

async function updatePixelCanvas() {
  const canvas = $("#pixel-canvas");
  const label = $("#pixel-canvas-label");
  const stage = sessionState?.stage;
  const fo = sessionState?.finalize_output;

  if (fo && fo.image_ref && fo.image_ref.startsWith("blob://")) {
    if (fo.image_ref !== lastRenderedImageRef) {
      lastRenderedImageRef = fo.image_ref;
      canvas.classList.add("forming");
      try {
        const img = new Image();
        const url = `/api/session/${sessionState.session_id}/render?t=${Date.now()}`;
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = url;
        });
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        // Next frame, so the browser has committed the sharp pixels
        // before the CSS blur transition starts easing out.
        requestAnimationFrame(() => canvas.classList.remove("forming"));
        label.textContent = `rendered by ${fo.renderer_used || "polish"}`;
      } catch {
        label.textContent = "render exists but couldn't be fetched";
      }
    }
    return;
  }

  if (fo && fo.image_ref) {
    // MockBackend's own `render://mock/...` refs (not a real blob) —
    // finalize genuinely succeeded, there's just no real image behind
    // it to fetch. Shown honestly as its own state, not as an error and
    // not silently indistinguishable from "no render yet."
    lastRenderedImageRef = fo.image_ref;
    canvas.classList.remove("forming");
    drawPlaceholderGrid(sessionState?.session_id, "idle");
    label.textContent = `${fo.renderer_used || "polish"} (mock — no real image)`;
    return;
  }

  lastRenderedImageRef = null;
  if (stage === "finalizing") {
    canvas.classList.add("forming");
    drawPlaceholderGrid(sessionState?.session_id, "forming");
    label.textContent = "polish tier forming the image…";
  } else {
    canvas.classList.remove("forming");
    drawPlaceholderGrid(sessionState?.session_id, "idle");
    label.textContent = stage === "sketching" || stage === "conversing"
      ? "no render yet — finalize when ready"
      : "no render yet";
  }
}

function renderChat() {
  const log = $("#chat-log");
  log.innerHTML = "";
  for (const turn of sessionState.conversation_history || []) {
    const div = document.createElement("div");
    div.className = `msg role-${turn.role}`;
    div.innerHTML = `<span class="who">${turn.role}</span><span>${escapeHtml(turn.content)}</span>`;
    log.appendChild(div);
  }
  log.scrollTop = log.scrollHeight;
}

function renderFinalizePanel() {
  const out = $("#finalize-panel");
  const fo = sessionState.finalize_output;
  if (!fo || !fo.image_ref) {
    out.innerHTML = `<p class="hint">No render yet — finalize to produce one.</p>`;
    return;
  }
  const scores = fo.verifier_scores || {};
  const rows = Object.entries(scores)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `
      <div class="score-label">${k.replace(/_/g, " ")}</div>
      <div>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${Math.round(v * 100)}%"></div></div>
      </div>
    `).join("");
  out.innerHTML = `
    <div><strong>Renderer:</strong> ${fo.renderer_used || "—"}</div>
    <div class="hint" style="margin-top:0.3rem">${escapeHtml(fo.image_ref)}</div>
    <div class="score-grid">${rows}</div>
  `;
}

function renderCritiquePanel() {
  // sessionState.critique (critique.result: critique_source, overall_score,
  // dimensions{name:{score,note}}, suggested_edits, raw_model_output_ref)
  // was fetched and stored by the "Critique" button's handler, but nothing
  // ever rendered it — clicking Critique gave the user zero visible
  // feedback that anything happened, let alone what the critique said.
  const out = $("#critique-panel");
  const critique = sessionState.critique;
  if (!critique || !critique.requested || !critique.result) {
    out.innerHTML = `<p class="hint">No critique yet — finalize first, then critique.</p>`;
    return;
  }
  const result = critique.result;
  const dimensionRows = Object.entries(result.dimensions || {})
    .map(([name, d]) => `
      <div class="score-label">${escapeHtml(name.replace(/_/g, " "))}</div>
      <div>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${Math.round((d.score || 0) * 100)}%"></div></div>
        ${d.note ? `<div class="hint" style="margin-top:0.15rem">${escapeHtml(d.note)}</div>` : ""}
      </div>
    `).join("");
  const editsList = (result.suggested_edits || [])
    .map((edit) => `<li>${escapeHtml(JSON.stringify(edit))}</li>`)
    .join("");
  out.innerHTML = `
    <div><strong>Source:</strong> ${escapeHtml(critique.source || result.critique_source || "—")}</div>
    <div><strong>Overall score:</strong> ${Math.round((result.overall_score || 0) * 100)}%</div>
    ${critique.timestamp ? `<div class="hint" style="margin-top:0.15rem">${escapeHtml(critique.timestamp)}</div>` : ""}
    <div class="score-grid" style="margin-top:0.5rem">${dimensionRows}</div>
    ${editsList ? `<div style="margin-top:0.5rem"><strong>Suggested edits:</strong><ul>${editsList}</ul></div>` : ""}
  `;
}

function applySession(data) {
  sessionState = data;
  $("#no-session").classList.add("hidden");
  $("#has-session").classList.remove("hidden");
  $("#session-id").textContent = data.session_id || "(unknown)";
  $("#session-rev-tag").textContent = `rev ${data.revision}`;
  renderPhasePipeline(data.stage);
  renderChat();
  renderFinalizePanel();
  renderCritiquePanel();
  updatePixelCanvas();
}

$("#btn-new-session").addEventListener("click", async () => {
  const r = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await r.json();
  if (r.ok) applySession(data);
});

// ------------------------------------------------------------------ //
// Chat/session UX: busy-state locking, error surfacing, a lightweight
// step-progress ticker for finalize. All three buttons + the message
// input share one `busy` flag — like the file said, only one request
// per session should ever be in flight from this UI at a time, since
// the backend itself serializes writes per session (StaleDesignStateError
// on a revision mismatch) and letting two fire at once would just
// surface that as a confusing error instead of preventing it up front.
// ------------------------------------------------------------------ //

let busy = false;

function setBusy(isBusy) {
  busy = isBusy;
  $("#message-input").disabled = isBusy;
  $("#btn-send").disabled = isBusy;
  $("#btn-finalize").disabled = isBusy;
  $("#btn-critique").disabled = isBusy;
}

function showChatError(message) {
  const box = $("#chat-error");
  box.textContent = message;
  box.classList.remove("hidden");
}
function clearChatError() {
  $("#chat-error").classList.add("hidden");
}

function addThinkingBubble(who) {
  const log = $("#chat-log");
  const div = document.createElement("div");
  div.className = "msg thinking";
  div.id = "thinking-bubble";
  div.innerHTML = `<span class="who">${who}</span><span>thinking…</span>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function removeThinkingBubble() {
  $("#thinking-bubble")?.remove();
}

// Z-Image-Turbo's own documented step count (polish_default_backend.py:
// num_inference_steps=9) — used here only to label the overlay
// truthfully ("step N/9"), not because this synchronous /finalize call
// reports real per-step progress back to the browser. The fill ticks on
// a fixed interval against a rough expected duration, explicitly
// labeled "estimated" in the UI text, and always completes to 100% the
// moment the real response arrives rather than drifting past it.
const POLISH_STEP_COUNT = 9;
let stepTicker = null;

function startStepOverlay() {
  const overlay = $("#pixel-step-overlay");
  const fill = $("#pixel-step-fill");
  const label = $("#pixel-step-label");
  overlay.classList.remove("hidden");
  let step = 0;
  const expectedMs = 6000; // rough — real duration depends on hardware, not measured here
  const tickMs = expectedMs / POLISH_STEP_COUNT;
  fill.style.width = "0%";
  label.textContent = `step 0/${POLISH_STEP_COUNT} (estimated)`;
  stepTicker = setInterval(() => {
    step = Math.min(step + 1, POLISH_STEP_COUNT - 1); // never claims 100% until the real response lands
    fill.style.width = `${(step / POLISH_STEP_COUNT) * 100}%`;
    label.textContent = `step ${step}/${POLISH_STEP_COUNT} (estimated)`;
  }, tickMs);
}
function stopStepOverlay(completed) {
  if (stepTicker) { clearInterval(stepTicker); stepTicker = null; }
  const overlay = $("#pixel-step-overlay");
  if (completed) {
    $("#pixel-step-fill").style.width = "100%";
    $("#pixel-step-label").textContent = `step ${POLISH_STEP_COUNT}/${POLISH_STEP_COUNT} — done`;
    setTimeout(() => overlay.classList.add("hidden"), 600);
  } else {
    overlay.classList.add("hidden");
  }
}

async function extractErrorDetail(r) {
  try {
    const data = await r.json();
    return data.detail || data.error || `request failed (${r.status})`;
  } catch {
    return `request failed (${r.status})`;
  }
}

$("#message-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!sessionState || busy) return;
  const input = $("#message-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  clearChatError();
  setBusy(true);
  addThinkingBubble("planner");
  try {
    const r = await fetch(`/api/session/${sessionState.session_id}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    removeThinkingBubble();
    if (r.ok) {
      applySession(await r.json());
    } else {
      showChatError(await extractErrorDetail(r));
    }
  } catch (err) {
    removeThinkingBubble();
    showChatError(`Couldn't reach the backend: ${err.message || err}`);
  } finally {
    setBusy(false);
    refreshStatus();
  }
});

$("#btn-finalize").addEventListener("click", async () => {
  if (!sessionState || busy) return;
  const quality = $("#finalize-quality").checked;
  clearChatError();
  setBusy(true);
  // Show the "forming" state immediately, before the (potentially
  // multi-second, real-model) response comes back — the whole point of
  // a pixel-forming visualization is showing the wait, not just the
  // result once it's already done.
  renderPhasePipeline("finalizing");
  $("#pixel-canvas").classList.add("forming");
  drawPlaceholderGrid(sessionState.session_id, "forming");
  $("#pixel-canvas-label").textContent = "polish tier forming the image…";
  startStepOverlay();

  try {
    const r = await fetch(`/api/session/${sessionState.session_id}/finalize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quality }),
    });
    if (r.ok) {
      stopStepOverlay(true);
      applySession(await r.json());
    } else {
      stopStepOverlay(false);
      renderPhasePipeline(sessionState.stage); // revert the optimistic "finalizing" node
      showChatError(await extractErrorDetail(r));
    }
  } catch (err) {
    stopStepOverlay(false);
    renderPhasePipeline(sessionState.stage);
    showChatError(`Couldn't reach the backend: ${err.message || err}`);
  } finally {
    setBusy(false);
    refreshStatus();
  }
});

$("#btn-critique").addEventListener("click", async () => {
  if (!sessionState || busy) return;
  clearChatError();
  setBusy(true);
  renderPhasePipeline("critiquing");
  try {
    const r = await fetch(`/api/session/${sessionState.session_id}/critique`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (r.ok) {
      applySession(await r.json());
    } else {
      renderPhasePipeline(sessionState.stage);
      showChatError(await extractErrorDetail(r));
    }
  } catch (err) {
    renderPhasePipeline(sessionState.stage);
    showChatError(`Couldn't reach the backend: ${err.message || err}`);
  } finally {
    setBusy(false);
    refreshStatus();
  }
});
