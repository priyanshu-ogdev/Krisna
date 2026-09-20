# Krisna Inference & Serving Architecture

Comprehensive reference for single-GPU serving, swap orchestration, and multi-turn agentic flows.

> [!IMPORTANT]
> **Canonical Specification**: Defined in [docs/PRD.md §5](file:///d:/Krisna/docs/PRD.md#5-data-model--core-flows) (Data Model & Core Flows) and [docs/PRD.md §7](file:///d:/Krisna/docs/PRD.md#7-inference-orchestration) (Inference Orchestration).
> Krisna is a **swap orchestrator** that never keeps more than one large generative tier GPU-resident at a time.

---

## 1. System Pipeline Architecture

```
User Intent (Chat Turn)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Tier A: Baseline (Always Resident, ~9.5GB VRAM)        │
│  - Planner Backend (Qwen3.5-9B, NF4, RAG-grounded)     │
│  - Sketch Backend (MaskGIT Transformer, 512px)         │
└────────────────────────────────────────────────────────┘
       │                                  │
       ▼ (User requests Finalize)         ▼ (User requests Critique)
┌────────────────────────────────┐  ┌────────────────────────────────┐
│ Tier B: Polish Tier (Swapped)  │  │ Tier C: Critic Tier (Swapped)  │
│  - Polish-Default (Z-Image)    │  │  - Gemma 4 31B Dense (NF4)     │
│  - or Polish-Quality (Qwen)    │  │  - Zero-shot VLM-as-a-judge    │
│  - Verifier Stack Gate         │  │  - Multi-axis Critique Output  │
└────────────────────────────────┘  └────────────────────────────────┘
       │                                  │
       └─────────────────┬────────────────┘
                         ▼
        Orchestrator unloads swappable tier,
        reloads Tier A baseline, returns to IDLE.
```

---

## 2. Residency State Machine (PRD §7.4)

Verified transition table implemented in `krisna_inference.orchestrator.state_machine`:

```mermaid
stateDiagram-v2
    [*] --> IDLE_RESIDENT
    IDLE_RESIDENT --> SWAPPING_TO_POLISH: Request Finalize
    SWAPPING_TO_POLISH --> POLISH_RESIDENT: Baseline Unloaded, Polish Loaded
    POLISH_RESIDENT --> SWAPPING_BACK_FROM_POLISH: Polish Render Complete
    SWAPPING_BACK_FROM_POLISH --> IDLE_RESIDENT: Polish Unloaded, Baseline Reloaded

    IDLE_RESIDENT --> SWAPPING_TO_CRITIC: Request Critique
    SWAPPING_TO_CRITIC --> CRITIC_RESIDENT: Baseline Unloaded, Critic Loaded
    CRITIC_RESIDENT --> SWAPPING_BACK_FROM_CRITIC: Critique Complete
    SWAPPING_BACK_FROM_CRITIC --> IDLE_RESIDENT: Critic Unloaded, Baseline Reloaded

    SWAPPING_TO_POLISH --> ERROR_RECOVERY: Load Failure (OOM)
    SWAPPING_TO_CRITIC --> ERROR_RECOVERY: Load Failure (OOM)
    ERROR_RECOVERY --> IDLE_RESIDENT: Re-admit Baseline, Session Preserved
```

---

## 3. Resource Ledgers & Envelopes (PRD §3, §7.2, §7.3)

Admission is deterministic and CI-testable, governed by `VRAMLedger` and `RAMLedger` against declared budgets:

| Tier | Full Envelope (24GB Target) | Low-VRAM Envelope (12GB Target) | Offload Mechanism |
|---|---|---|---|
| **Planner** (Always Resident) | 6.5 GB VRAM / 0 GB RAM | 6.5 GB VRAM / 0 GB RAM | BitsAndBytes NF4 |
| **Sketch** (Always Resident) | 3.0 GB VRAM / 0 GB RAM | 3.0 GB VRAM / 0 GB RAM | PyTorch BF16 / FP32 |
| **Polish-Default** (Z-Image-Turbo) | 14.0 GB VRAM / 0 GB RAM | 12.0 GB VRAM / 2.0 GB RAM | `enable_model_cpu_offload()` |
| **Polish-Quality** (Qwen-Image-Edit) | 16.0 GB VRAM / 0 GB RAM | 10.0 GB VRAM / 10.0 GB RAM | `enable_model_cpu_offload()` |
| **Critic** (Gemma 4 31B) | 18.0 GB VRAM / 0 GB RAM | 11.5 GB VRAM / 45.0 GB RAM | Transformers + bnb FP32 offload |

---

## 4. Multi-Turn Agentic Feedback Loop

1. **Context Preservation**:
   - Every `POST /session/{id}/message` carries cumulative `conversation_history` into the Planner backend.
2. **Dynamic Constraints Merging**:
   - Structured JSON output appended by Planner is parsed and stripped from the user reply.
   - `constraint_updates` (theme, palette, devices, and `locked_regions: [{bbox, reason}]`) are merged into `DesignState.constraints`.
3. **Conditioned Generation & Prompt Synthesis**:
   - Sketch generation tokens (`vq_tokens`) ground CLIP text embeddings in active constraints.
   - Finalize synthesizes a complete, detailed prompt from active constraints when none is explicitly supplied.
4. **Closed Critic Feedback**:
   - Critic passes return structured `suggested_edits` with bounding boxes.
   - The Web Studio UI provides a 1-click **"✨ Iterate with Critic Edits"** button feeding structured feedback directly into the next Planner turn.

---

## 5. Production Docker Containerization & Dual-Venv Isolation

For release deployments, Krisna Inference is packaged into a high-performance GPU container architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ docker/Dockerfile.inference (CUDA 12.4.1 / Python 3.11)     │
│                                                             │
│  ┌─────────────────────────┐   ┌──────────────────────────┐ │
│  │ /opt/venv-inference     │   │ /opt/venv-critic         │ │
│  │ - torch >= 2.6.0        │   │ - torch >= 2.6.0         │ │
│  │ - transformers >= 5.2.0 │   │ - transformers == 5.5.0  │ │
│  │ - diffusers >= 0.31.0   │   │ - unsloth + unsloth_zoo  │ │
│  │ - SwapOrchestrator      │   │ - bitsandbytes           │ │
│  │ - Planner / Sketch /    │   │ - critic_worker.py       │ │
│  │   Polish models         │   │                          │ │
│  └────────────┬────────────┘   └─────────────▲────────────┘ │
│               │                              │              │
│               └──────── JSON IPC Pipe ───────┘              │
│                  (KRISNA_CRITIC_VENV_PYTHON)                │
└─────────────────────────────────────────────────────────────┘
```

- **Container Compose Stack**: `docker compose up -d` launches both the GPU inference engine (`:8420`) and the Web Studio UI (`:3000`).
- **Host Launchers**: `./scripts/docker/run_docker.sh` and `.\scripts\docker\run_docker.ps1` execute host preflight probes (Docker daemon, NVIDIA Container Toolkit, and WSL2 acceleration).

---

## 6. Fail-Fast Hardware & Deployment Preflight Verification

Implemented in `krisna_inference.common.hardware` and runnable via `scripts/inference/check_hardware.py`:
- Probes GPU existence, CUDA runtime, compute capability (`sm_75+` required, `sm_80+` recommended for native BF16), and physical VRAM.
- Enforces envelopes: 16GB+ default, 11.5GB+ low-VRAM mode (`KRISNA_LOW_VRAM_MODE=1`).
- Startup guard: When `KRISNA_USE_REAL_BACKENDS=1`, `service.py` immediately aborts if hardware is incompatible, preventing runtime hangs or crashes. MockBackend is strictly reserved for CI and automated testing.

---

## 7. REST API Endpoints

FastAPI service exposed on default port `8420`:

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` / `/healthz` | Liveness & readiness probes for Docker/k8s (residency state and VRAM snapshot). |
| `GET` | `/hardware` | Live hardware telemetry (GPU, driver, CUDA, free/total VRAM, and config status). |
| `POST` | `/session` | Initializes a new `DesignState` session. |
| `GET` | `/session/{id}` | Retrieves active `DesignState` snapshot. |
| `POST` | `/session/{id}/message` | Advances conversational turn (Planner + Sketch pass). |
| `POST` | `/session/{id}/finalize` | Initiates tier swap, decodes VQ tokens, runs Polish, scores verifiers. |
| `POST` | `/session/{id}/critique` | Initiates tier swap to Gemma 4, evaluates design, returns structured feedback. |
| `GET` | `/session/{id}/render` | Streams raw finalized image bytes (`image/png`). |
| `GET` | `/orchestrator/status` | Reports residency state, declared ledgers, and live hardware probes. |

---

## 8. Related Documentation

- **[inference/README.md](file:///d:/Krisna/inference/README.md)**: Serving setup, mock vs real backends, Docker flags, and requirements.
- **[inference/frontend/README.md](file:///d:/Krisna/inference/frontend/README.md)**: Node.js Web Studio & Canvas2D Generalization Visualizer.
- **[inference/runtime/README.md](file:///d:/Krisna/inference/runtime/README.md)**: CLI harness for multi-turn manual testing.
- **[docs/review/28_docker_isolation_and_hardware_preflight.md](file:///d:/Krisna/docs/review/28_docker_isolation_and_hardware_preflight.md)**: Phase 28 research and verification report.

