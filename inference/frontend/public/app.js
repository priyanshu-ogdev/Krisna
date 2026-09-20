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
// Health pill & System status
// ------------------------------------------------------------------ //
async function refreshHealth() {
  const pill = $("#health-pill");
  const pillText = $("#health-pill-text") || pill;
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    pillText.textContent = data.residency_state;
    pill.className = "pill ok";
  } catch (e) {
    pillText.textContent = "backend unreachable";
    pill.className = "pill err";
  }
}
refreshHealth();
setInterval(refreshHealth, 5000);

async function loadSystemInfo() {
  try {
    const r = await fetch("/api/system/info");
    if (r.ok) {
      const data = await r.json();
      if ($("#hw-pill-text")) {
        $("#hw-pill-text").textContent = `${data.targetGpu.split(" ")[1] || "A6000"} · ${data.targetRam}`;
      }
    }
  } catch {}
}
loadSystemInfo();

// ------------------------------------------------------------------ //
// Checkpoints auto-scanner
// ------------------------------------------------------------------ //
let discoveredCheckpoints = null;

async function loadCheckpoints() {
  const container = $("#checkpoints-list");
  if (!container) return;
  try {
    const r = await fetch("/api/checkpoints/scan");
    if (!r.ok) throw new Error("scan failed");
    const data = await r.json();
    discoveredCheckpoints = data;
    renderCheckpoints(data);
  } catch (e) {
    container.innerHTML = `<div class="hint">Checkpoints scan unavailable (${escapeHtml(e.message)})</div>`;
  }
}

function renderCheckpoints(data) {
  const container = $("#checkpoints-list");
  if (!container) return;
  const items = [];

  // Sketch
  if (data.sketch && data.sketch.length > 0) {
    for (const s of data.sketch) {
      items.push(`
        <div class="ckpt-card">
          <div class="ckpt-info">
            <div class="ckpt-title-row">
              <span class="ckpt-name">Sketch (${s.isFinal ? "Final" : "Step"})</span>
              <span class="ckpt-badge ready">${s.sizeMb}MB</span>
            </div>
            <span class="ckpt-path">${escapeHtml(s.path)}</span>
          </div>
          <button type="button" class="btn btn-ghost btn-xs" onclick="useSketchPath('${escapeHtml(s.path)}')">Use</button>
        </div>
      `);
    }
  } else {
    items.push(`
      <div class="ckpt-card">
        <div class="ckpt-info">
          <div class="ckpt-title-row">
            <span class="ckpt-name">Sketch Tier</span>
            <span class="ckpt-badge missing">Not Found</span>
          </div>
          <span class="ckpt-path">checkpoints/sketch_stage2_512/checkpoint_final.pt</span>
        </div>
      </div>
    `);
  }

  // Polish
  if (data.polish && data.polish.length > 0) {
    for (const p of data.polish) {
      items.push(`
        <div class="ckpt-card">
          <div class="ckpt-info">
            <div class="ckpt-title-row">
              <span class="ckpt-name">${escapeHtml(p.name)}</span>
              <span class="ckpt-badge ready">${p.type}</span>
            </div>
            <span class="ckpt-path">${escapeHtml(p.path)}</span>
          </div>
          <button type="button" class="btn btn-ghost btn-xs" onclick="usePolishPath('${escapeHtml(p.path)}')">Use</button>
        </div>
      `);
    }
  } else {
    items.push(`
      <div class="ckpt-card">
        <div class="ckpt-info">
          <div class="ckpt-title-row">
            <span class="ckpt-name">Polish LoRA</span>
            <span class="ckpt-badge missing">Not Found</span>
          </div>
          <span class="ckpt-path">models/dpo_checkpoints/**/final/ or checkpoints/polish_default_lora</span>
        </div>
      </div>
    `);
  }

  // Critic venv
  items.push(`
    <div class="ckpt-card">
      <div class="ckpt-info">
        <div class="ckpt-title-row">
          <span class="ckpt-name">Critic Environment</span>
          <span class="ckpt-badge ${data.critic && data.critic.ready ? "ready" : "missing"}">${data.critic && data.critic.ready ? "Ready" : "Missing"}</span>
        </div>
        <span class="ckpt-path">${escapeHtml((data.critic && data.critic.venv) || "venv-critic/Scripts/python.exe")}</span>
      </div>
    </div>
  `);

  container.innerHTML = items.join("");
}

window.useSketchPath = function(p) {
  const input = $("#input-sketch-ckpt");
  if (input) input.value = p;
};
window.usePolishPath = function(p) {
  const input = $("#input-polish-lora");
  if (input) input.value = p;
};

function applyBestDiscoveredCheckpoints() {
  if (!discoveredCheckpoints) return;
  if (discoveredCheckpoints.sketch && discoveredCheckpoints.sketch.length > 0) {
    const best = discoveredCheckpoints.sketch.find(s => s.isFinal) || discoveredCheckpoints.sketch[0];
    useSketchPath(best.path);
  }
  if (discoveredCheckpoints.polish && discoveredCheckpoints.polish.length > 0) {
    const best = discoveredCheckpoints.polish.find(p => p.type === "dpo") || discoveredCheckpoints.polish[0];
    usePolishPath(best.path);
  }
}
loadCheckpoints();


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

    const modePill = $("#mode-pill");
    const modeText = $("#mode-pill-text") || modePill;
    if (modePill) {
      if (data.real_vram) {
        modePill.className = "pill mode-pill real";
        modeText.textContent = data.low_vram_mode ? "Real GPU (Low-VRAM)" : "Real GPU Mode";
      } else {
        modePill.className = "pill mode-pill mock";
        modeText.textContent = "MockBackend Mode";
      }
    }
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

// ==================================================================== //
// Generalization & Flow-Matching Canvas Engine
//
// Models the neural trajectory from latent noise -> semantic layout
// -> component crystallization -> high-frequency polish convergence
// using hardware-accelerated 60fps Canvas2D rendering.
// ==================================================================== //

class GeneralizationEngine {
  constructor() {
    this.canvas = $("#pixel-canvas");
    this.ctx = this.canvas?.getContext("2d");
    this.wireframeCanvas = $("#wireframe-canvas");
    this.wfCtx = this.wireframeCanvas?.getContext("2d");
    this.viewMode = "pixels"; // "pixels" | "wireframe" | "flow"

    this.animating = false;
    this.animFrameId = null;
    this.startTime = 0;
    this.durationMs = 6000;
    this.stepCount = 9;
    this.progress = 0;
    this.cachedImage = null;

    this.particles = [];
    this.initParticles(35);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden && this.animating) {
        this.pause();
      } else if (!document.hidden && this.animating) {
        this.resume();
      }
    });
  }

  initParticles(count) {
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * 280,
        y: Math.random() * 280,
        vx: (Math.random() - 0.5) * 1.5,
        vy: (Math.random() - 0.5) * 1.5,
        size: Math.random() * 2.5 + 1,
        color: Math.random() > 0.5 ? "rgba(99, 102, 241, " : "rgba(6, 182, 212, ",
        alpha: Math.random() * 0.7 + 0.3,
      });
    }
  }

  setViewMode(mode) {
    this.viewMode = mode;
    $$(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    if (this.wireframeCanvas) {
      this.wireframeCanvas.classList.toggle("hidden", mode !== "wireframe");
    }
    this.redraw();
  }

  startDenoising(expectedDurationMs = 6000, stepCount = 9) {
    this.durationMs = expectedDurationMs;
    this.stepCount = stepCount;
    this.startTime = performance.now();
    this.animating = true;
    this.progress = 0;
    this.cachedImage = null;

    if (this.wireframeCanvas) this.wireframeCanvas.classList.add("hidden");
    this.loop();
  }

  stopDenoising(completed = true) {
    this.animating = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (completed) {
      this.progress = 1.0;
      this.updateTelemetry(1.0, this.stepCount);
    }
  }

  pause() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  resume() {
    if (this.animating && !this.animFrameId) {
      this.loop();
    }
  }

  loop() {
    if (!this.animating) return;
    const now = performance.now();
    const elapsed = now - this.startTime;
    this.progress = Math.min(0.96, (elapsed / this.durationMs) * 0.95);

    this.renderDenoisingFrame(this.progress);
    this.updateTelemetry(this.progress, this.stepCount);

    this.animFrameId = requestAnimationFrame(() => this.loop());
  }

  updateTelemetry(t, stepCount) {
    const stepEl = $("#hud-step");
    const phaseEl = $("#hud-phase");
    const entropyEl = $("#hud-entropy");

    const activeStep = Math.min(stepCount, Math.floor(t * stepCount) + 1);
    if (stepEl) stepEl.textContent = `${activeStep}/${stepCount}`;

    let phaseName = "Latent Noise";
    if (t > 0.85) phaseName = "Pixel Convergence";
    else if (t > 0.55) phaseName = "Component Geometry";
    else if (t > 0.25) phaseName = "Semantic Manifold";
    if (phaseEl) phaseEl.textContent = phaseName;

    if (entropyEl) {
      const entropy = Math.max(0.02, (1 - t) * 0.98 + (Math.sin(performance.now() * 0.01) * 0.02));
      entropyEl.textContent = entropy.toFixed(2);
    }
  }

  renderDenoisingFrame(t) {
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, w, h);

    if (this.viewMode === "flow") {
      this.drawLatentFlowField(ctx, w, h, t);
    } else {
      this.drawGeneralizationStage(ctx, w, h, t);
    }
  }

  drawLatentFlowField(ctx, w, h, t) {
    const cols = 14;
    const cellW = w / cols;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellW, 0);
      ctx.lineTo(i * cellW, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellW);
      ctx.lineTo(w, i * cellW);
      ctx.stroke();
    }

    const time = performance.now() * 0.002;
    for (const p of this.particles) {
      p.x += p.vx * (1.5 - t);
      p.y += p.vy * (1.5 - t);
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.fillStyle = `${p.color}${(1 - t * 0.5) * p.alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * (1 - t * 0.4), 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.rotate(time * 0.5);
    ctx.strokeStyle = `rgba(245, 158, 11, ${0.4 + t * 0.5})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 35 * (1 - t * 0.5), 0, Math.PI * (1 + t));
    ctx.stroke();
    ctx.restore();
  }

  drawGeneralizationStage(ctx, w, h, t) {
    const manifoldAlpha = Math.min(1.0, t * 1.8);
    ctx.fillStyle = `rgba(14, 19, 27, ${manifoldAlpha * 0.9})`;
    ctx.fillRect(16, 16, w - 32, h - 32);

    ctx.fillStyle = `rgba(99, 102, 241, ${manifoldAlpha * 0.18})`;
    ctx.fillRect(24, 24, w - 48, 36);

    ctx.fillStyle = `rgba(6, 182, 212, ${manifoldAlpha * 0.15})`;
    ctx.fillRect(24, 68, w - 48, 72);

    ctx.fillStyle = `rgba(255, 255, 255, ${manifoldAlpha * 0.05})`;
    ctx.fillRect(24, 148, w - 48, 28);
    ctx.fillRect(24, 182, w - 48, 28);

    ctx.fillStyle = `rgba(245, 158, 11, ${manifoldAlpha * 0.25})`;
    ctx.fillRect(24, 218, w - 48, 34);

    const noiseOpacity = Math.max(0, 0.85 - t * 0.9);
    if (noiseOpacity > 0.01) {
      const grid = 14;
      const cell = w / grid;
      const timeSeed = Math.floor(performance.now() / 80);
      for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
          const rand = seededRand(timeSeed + x * 17.3 + y * 31.7);
          if (rand > 0.4) {
            ctx.fillStyle = `rgba(${rand > 0.7 ? "245,158,11" : "99,102,241"}, ${noiseOpacity * rand * 0.4})`;
            ctx.fillRect(x * cell, y * cell, cell, cell);
          }
        }
      }
    }

    if (t > 0.45) {
      const wfAlpha = Math.min(1.0, (t - 0.45) / 0.4);
      ctx.strokeStyle = `rgba(6, 182, 212, ${wfAlpha * 0.6})`;
      ctx.lineWidth = 1;
      ctx.strokeRect(24, 24, w - 48, 36);
      ctx.strokeRect(24, 68, w - 48, 72);
      ctx.strokeRect(24, 148, w - 48, 28);
      ctx.strokeRect(24, 182, w - 48, 28);
      ctx.strokeRect(24, 218, w - 48, 34);
    }
  }

  drawAmbientGrid(sessionId) {
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, w, h);

    const grid = 14;
    const cell = w / grid;
    const seedBase = [...(sessionId || "krisna")].reduce((a, c) => a + c.charCodeAt(0), 0);

    for (let y = 0; y < grid; y++) {
      for (let x = 0; x < grid; x++) {
        const r = seededRand(seedBase + x * 13.1 + y * 7.7);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.015 + r * 0.035})`;
        ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
      }
    }

    ctx.strokeStyle = "rgba(99, 102, 241, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2 - 15, h / 2);
    ctx.lineTo(w / 2 + 15, h / 2);
    ctx.moveTo(w / 2, h / 2 - 15);
    ctx.lineTo(w / 2 + 15, h / 2);
    ctx.stroke();

    if (this.wireframeCanvas) {
      this.wireframeCanvas.classList.add("hidden");
    }
  }

  drawLoadedImage(img) {
    this.cachedImage = img;
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.drawImage(img, 0, 0, w, h);

    this.updateTelemetry(1.0, this.stepCount);

    if (this.viewMode === "wireframe") {
      this.renderWireframe(this.wireframeCanvas, this.wfCtx, w, h);
    } else if (this.wireframeCanvas) {
      this.wireframeCanvas.classList.add("hidden");
    }
  }

  renderWireframe(canvas, ctx, w, h) {
    if (!canvas || !ctx) return;
    canvas.classList.remove("hidden");
    ctx.clearRect(0, 0, w, h);

    const scaleX = w / 280;
    const scaleY = h / 280;

    const components = [
      { label: "Nav: Header / Brand", x: 20, y: 16, w: 240, h: 36, color: "#6366f1" },
      { label: "Hero: Balance Card", x: 20, y: 60, w: 240, h: 74, color: "#06b6d4" },
      { label: "Row: Quick Actions", x: 20, y: 142, w: 240, h: 32, color: "#10b981" },
      { label: "List: Activity Data", x: 20, y: 182, w: 240, h: 42, color: "#8b5cf6" },
      { label: "Button: Primary CTA", x: 20, y: 232, w: 240, h: 34, color: "#f59e0b" },
    ];

    for (const comp of components) {
      const rx = comp.x * scaleX;
      const ry = comp.y * scaleY;
      const rw = comp.w * scaleX;
      const rh = comp.h * scaleY;

      ctx.fillStyle = `${comp.color}15`;
      ctx.fillRect(rx, ry, rw, rh);

      ctx.strokeStyle = comp.color;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(rx, ry, rw, rh);

      ctx.fillStyle = "rgba(8, 10, 15, 0.85)";
      ctx.fillRect(rx + 2, ry + 2, 95 * scaleX, 13 * scaleY);
      ctx.fillStyle = comp.color;
      ctx.font = `${Math.round(8 * scaleX)}px "JetBrains Mono", monospace`;
      ctx.fillText(comp.label, rx + 4, ry + 10 * scaleY);
    }
  }

  redraw() {
    if (this.cachedImage) {
      this.drawLoadedImage(this.cachedImage);
    } else if (sessionState?.stage === "finalizing") {
      this.renderDenoisingFrame(this.progress);
    } else {
      this.drawAmbientGrid(sessionState?.session_id);
    }
  }

  startRadarScan() {
    const radar = $("#radar-scan-line");
    if (radar) radar.classList.remove("hidden");
  }

  stopRadarScan() {
    const radar = $("#radar-scan-line");
    if (radar) radar.classList.add("hidden");
  }
}

const genEngine = new GeneralizationEngine();

// Backward compatibility helper
function drawPlaceholderGrid(sessionId, tone) {
  if (tone === "forming") {
    genEngine.startDenoising();
  } else {
    genEngine.drawAmbientGrid(sessionId);
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
        genEngine.stopDenoising(true);
        genEngine.drawLoadedImage(img);
        requestAnimationFrame(() => canvas.classList.remove("forming"));
        label.textContent = `rendered by ${fo.renderer_used || "polish"}`;
      } catch {
        label.textContent = "render exists but couldn't be fetched";
      }
    }
    return;
  }

  if (fo && fo.image_ref) {
    lastRenderedImageRef = fo.image_ref;
    canvas.classList.remove("forming");
    genEngine.stopDenoising(true);
    genEngine.drawAmbientGrid(sessionState?.session_id);
    label.textContent = `${fo.renderer_used || "polish"} (mock — no real image)`;
    return;
  }

  lastRenderedImageRef = null;
  if (stage === "finalizing") {
    canvas.classList.add("forming");
    label.textContent = "polish tier forming the image…";
  } else {
    canvas.classList.remove("forming");
    genEngine.stopDenoising(false);
    genEngine.drawAmbientGrid(sessionState?.session_id);
    label.textContent = stage === "sketching" || stage === "conversing"
      ? "no render yet — finalize when ready"
      : "no render yet";
  }
}

function renderChat() {
  const log = $("#chat-log");
  log.innerHTML = "";

  // Active constraints header chip bar if defined
  const constraints = sessionState.constraints || {};
  if (constraints.style || (constraints.palette && constraints.palette.length > 0) || constraints.layout_hints) {
    const bar = document.createElement("div");
    bar.className = "active-constraints-bar";
    const parts = [];
    if (constraints.style) {
      parts.push(`<span class="constraint-chip">🎨 ${escapeHtml(constraints.style)}</span>`);
    }
    if (constraints.palette && constraints.palette.length > 0) {
      const swatches = constraints.palette.map(c => `<span class="color-dot" style="background:${escapeHtml(c)}" title="${escapeHtml(c)}"></span>`).join("");
      parts.push(`<span class="constraint-chip"><span class="swatches-inline">${swatches}</span> Palette (${constraints.palette.length})</span>`);
    }
    if (constraints.layout_hints) {
      parts.push(`<span class="constraint-chip">📐 ${escapeHtml(constraints.layout_hints)}</span>`);
    }
    bar.innerHTML = `<span class="constraints-title">Active Constraints:</span> ${parts.join(" ")}`;
    log.appendChild(bar);
  }

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
    out.innerHTML = `<p class="hint">No render produced yet — click <strong>Finalize Render</strong> above.</p>`;
    return;
  }
  const scores = fo.verifier_scores || {};
  const scoreEntries = Object.entries(scores).filter(([, v]) => v !== null && v !== undefined);

  let verifierCardsHtml = "";
  if (scoreEntries.length > 0) {
    verifierCardsHtml = `
      <div class="verifier-grid">
        ${scoreEntries.map(([k, v]) => `
          <div class="verifier-card">
            <div class="verifier-card-head">
              <span class="verifier-name">${escapeHtml(k.replace(/_/g, " "))}</span>
              <span class="verifier-pct">${Math.round(v * 100)}%</span>
            </div>
            <div class="verifier-track">
              <div class="verifier-fill" style="width: ${Math.round(v * 100)}%"></div>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  }

  out.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
      <div><strong>Renderer:</strong> <span class="tag tag-phase">${escapeHtml(fo.renderer_used || "polish")}</span></div>
      <div class="btn-row-sm">
        <button type="button" class="btn btn-ghost btn-xs" id="btn-zoom-inline">🔍 Inspect Fullscreen</button>
      </div>
    </div>
    <div class="hint" style="margin-top:0.2rem"><strong>Image Ref:</strong> <code>${escapeHtml(fo.image_ref)}</code></div>
    ${verifierCardsHtml}
  `;

  $("#btn-zoom-inline")?.addEventListener("click", openLightbox);
}

function renderCritiquePanel() {
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

  const suggestedEdits = result.suggested_edits || [];
  let editsList = "";
  if (suggestedEdits.length > 0) {
    editsList = suggestedEdits.map((edit, idx) => {
      if (typeof edit === "object" && edit !== null) {
        const region = edit.region ? `<span class="edit-region-badge">${escapeHtml(Array.isArray(edit.region) ? edit.region.join(", ") : String(edit.region))}</span>` : "";
        const instr = escapeHtml(edit.instruction || JSON.stringify(edit));
        return `<li class="critique-edit-item">
          ${region}
          <span class="edit-instruction">${instr}</span>
          <button type="button" class="btn btn-ghost btn-xs btn-apply-single-edit" data-edit-index="${idx}" title="Copy to prompt input">Apply ↳</button>
        </li>`;
      }
      return `<li class="critique-edit-item"><span class="edit-instruction">${escapeHtml(String(edit))}</span></li>`;
    }).join("");
  }

  const iterateBtnHtml = suggestedEdits.length > 0 ? `
    <div class="critique-action-bar">
      <button type="button" class="btn btn-primary btn-sm" id="btn-apply-all-edits">
        ✨ Iterate with Critic Edits
      </button>
      <span class="hint" style="margin-left: 0.5rem; align-self: center;">Preloads all critic edits into turn input</span>
    </div>
  ` : "";

  out.innerHTML = `
    <div><strong>Source:</strong> ${escapeHtml(critique.source || result.critique_source || "—")}</div>
    <div><strong>Overall score:</strong> ${Math.round((result.overall_score || 0) * 100)}%</div>
    ${critique.timestamp ? `<div class="hint" style="margin-top:0.15rem">${escapeHtml(critique.timestamp)}</div>` : ""}
    <div class="score-grid" style="margin-top:0.5rem">${dimensionRows}</div>
    ${editsList ? `<div style="margin-top:0.75rem"><strong>Suggested edits:</strong><ul class="critique-edits-list" style="margin-top:0.35rem">${editsList}</ul>${iterateBtnHtml}</div>` : ""}
  `;

  $("#btn-apply-all-edits")?.addEventListener("click", () => {
    const instructions = suggestedEdits.map(e => (typeof e === "object" && e?.instruction) ? e.instruction : String(e));
    const combinedPrompt = `Please refine the design addressing critic feedback: ${instructions.join("; ")}`;
    const input = $("#message-input");
    if (input) {
      input.value = combinedPrompt;
      input.focus();
      input.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });

  $$(".btn-apply-single-edit").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.editIndex, 10);
      const edit = suggestedEdits[idx];
      const instr = (typeof edit === "object" && edit?.instruction) ? edit.instruction : String(edit);
      const input = $("#message-input");
      if (input) {
        input.value = input.value ? `${input.value}; ${instr}` : `Address feedback: ${instr}`;
        input.focus();
        input.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });
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
  const qualityMode = $("#finalize-quality-mode")?.value || "default";
  const quality = qualityMode === "quality";
  const seedVal = $("#seed-input")?.value?.trim();
  const seed = seedVal ? parseInt(seedVal, 10) : undefined;

  clearChatError();
  setBusy(true);
  renderPhasePipeline("finalizing");
  $("#pixel-canvas").classList.add("forming");
  $("#pixel-canvas-label").textContent = `${quality ? "qwen-image-edit-2511" : "z-image-turbo"} forming pixels…`;

  // Start 60fps hardware-accelerated flow-matching denoising visualizer
  genEngine.startDenoising(quality ? 12000 : 6000, quality ? 28 : 9);
  startStepOverlay();

  try {
    const payload = { quality };
    if (seed !== undefined && !isNaN(seed)) payload.seed = seed;
    const r = await fetch(`/api/session/${sessionState.session_id}/finalize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (r.ok) {
      stopStepOverlay(true);
      applySession(await r.json());
    } else {
      stopStepOverlay(false);
      genEngine.stopDenoising(false);
      renderPhasePipeline(sessionState.stage);
      showChatError(await extractErrorDetail(r));
    }
  } catch (err) {
    stopStepOverlay(false);
    genEngine.stopDenoising(false);
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
  genEngine.startRadarScan();

  try {
    const r = await fetch(`/api/session/${sessionState.session_id}/critique`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    genEngine.stopRadarScan();
    if (r.ok) {
      applySession(await r.json());
    } else {
      renderPhasePipeline(sessionState.stage);
      showChatError(await extractErrorDetail(r));
    }
  } catch (err) {
    genEngine.stopRadarScan();
    renderPhasePipeline(sessionState.stage);
    showChatError(`Couldn't reach the backend: ${err.message || err}`);
  } finally {
    setBusy(false);
    refreshStatus();
  }
});

// ------------------------------------------------------------------ //
// Canvas View Mode Switcher (Pixels / Wireframe / Flow Latents)
// ------------------------------------------------------------------ //
$$(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    genEngine.setViewMode(btn.dataset.mode);
  });
});

// ------------------------------------------------------------------ //
// Archetype Quick-Start Chips
// ------------------------------------------------------------------ //
$$(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const prompt = chip.dataset.prompt;
    const input = $("#message-input");
    if (input && prompt) {
      input.value = prompt;
      input.focus();
    }
  });
});

// Randomize seed button
$("#btn-random-seed")?.addEventListener("click", () => {
  const seedInput = $("#seed-input");
  if (seedInput) {
    seedInput.value = Math.floor(Math.random() * 900000) + 100000;
  }
});

// Checkpoint discovery buttons
$("#btn-rescan-checkpoints")?.addEventListener("click", () => {
  loadCheckpoints();
});
$("#btn-apply-checkpoints")?.addEventListener("click", () => {
  applyBestDiscoveredCheckpoints();
});

// ------------------------------------------------------------------ //
// Full-Resolution Lightbox Modal & Trajectory Scrubber
// ------------------------------------------------------------------ //
function openLightbox() {
  if (!sessionState?.session_id) return;
  const modal = $("#lightbox-modal");
  const img = $("#lightbox-img");
  const meta = $("#lightbox-meta");
  if (!modal || !img) return;

  img.src = `/api/session/${sessionState.session_id}/render?t=${Date.now()}`;
  img.style.transform = "scale(1)";
  img.style.filter = "none";
  img.style.opacity = "1";

  // Reset scrubber
  const slider = $("#generalization-slider");
  if (slider) slider.value = 100;
  updateLightboxScrubber(1.0);

  if (meta) {
    const renderer = sessionState.finalize_output?.renderer_used || "polish";
    meta.textContent = `${renderer} · rev ${sessionState.revision}`;
  }
  modal.classList.remove("hidden");
}

function closeLightbox() {
  $("#lightbox-modal")?.classList.add("hidden");
}

function updateLightboxScrubber(t) {
  const label = $("#scrubber-stage-name");
  const img = $("#lightbox-img");
  if (!img) return;

  if (t >= 0.95) {
    if (label) label.textContent = `t = ${t.toFixed(2)} · Converged Polish Output`;
    img.style.filter = "none";
    img.style.opacity = "1";
  } else if (t >= 0.65) {
    if (label) label.textContent = `t = ${t.toFixed(2)} · Component Geometry Crystallization`;
    img.style.filter = `blur(${Math.round((1 - t) * 6)}px)`;
    img.style.opacity = `${0.65 + t * 0.35}`;
  } else if (t >= 0.25) {
    if (label) label.textContent = `t = ${t.toFixed(2)} · Semantic Manifold Formation`;
    img.style.filter = `blur(${Math.round((1 - t) * 12)}px) contrast(${0.6 + t * 0.4})`;
    img.style.opacity = `${0.35 + t * 0.5}`;
  } else {
    if (label) label.textContent = `t = ${t.toFixed(2)} · Latent Gaussian Noise`;
    img.style.filter = `blur(16px) saturate(${t * 0.5})`;
    img.style.opacity = `${0.15 + t * 0.5}`;
  }
}

$("#generalization-slider")?.addEventListener("input", (e) => {
  const t = parseInt(e.target.value, 10) / 100;
  updateLightboxScrubber(t);
});

$$(".milestone").forEach((m) => {
  m.addEventListener("click", () => {
    const t = parseInt(m.dataset.t, 10) / 100;
    const slider = $("#generalization-slider");
    if (slider) slider.value = t * 100;
    updateLightboxScrubber(t);
  });
});

let lightboxWireframeVisible = false;
$("#lightbox-toggle-wireframe")?.addEventListener("click", () => {
  const canvas = $("#lightbox-wireframe-canvas");
  const img = $("#lightbox-img");
  if (!canvas || !img) return;
  lightboxWireframeVisible = !lightboxWireframeVisible;
  canvas.classList.toggle("hidden", !lightboxWireframeVisible);
  if (lightboxWireframeVisible) {
    canvas.width = img.clientWidth || 512;
    canvas.height = img.clientHeight || 512;
    canvas.style.width = `${img.clientWidth}px`;
    canvas.style.height = `${img.clientHeight}px`;
    const ctx = canvas.getContext("2d");
    genEngine.renderWireframe(canvas, ctx, canvas.width, canvas.height);
  }
});

$("#pixel-canvas-wrap")?.addEventListener("click", () => {
  if (sessionState?.finalize_output?.image_ref) {
    openLightbox();
  }
});

$("#lightbox-close")?.addEventListener("click", closeLightbox);
$("#lightbox-backdrop")?.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#lightbox-modal")?.classList.contains("hidden")) {
    closeLightbox();
  }
});

$("#lightbox-zoom-fit")?.addEventListener("click", () => {
  const img = $("#lightbox-img");
  if (img) { img.style.transform = "scale(1)"; img.style.maxHeight = "65vh"; }
});
$("#lightbox-zoom-100")?.addEventListener("click", () => {
  const img = $("#lightbox-img");
  if (img) { img.style.transform = "scale(1)"; img.style.maxHeight = "none"; }
});
$("#lightbox-zoom-200")?.addEventListener("click", () => {
  const img = $("#lightbox-img");
  if (img) { img.style.transform = "scale(2)"; img.style.maxHeight = "none"; }
});

// ------------------------------------------------------------------ //
// Download & Export Handlers
// ------------------------------------------------------------------ //
function downloadRenderPng() {
  if (!sessionState?.session_id) return;
  const a = document.createElement("a");
  a.href = `/api/session/${sessionState.session_id}/render`;
  a.download = `krisna-render-${sessionState.session_id}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function exportDesignStateJson() {
  if (!sessionState) return;
  const blob = new Blob([JSON.stringify(sessionState, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `krisna-design-state-${sessionState.session_id || "session"}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

$("#btn-download-render")?.addEventListener("click", downloadRenderPng);
$("#lightbox-download")?.addEventListener("click", downloadRenderPng);
$("#btn-export-json")?.addEventListener("click", exportDesignStateJson);


