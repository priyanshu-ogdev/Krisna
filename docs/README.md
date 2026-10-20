# Documentation Index

> [!IMPORTANT]
> **Canonical Product & Research Specification**: [PRD.md](PRD.md)
> The authoritative single source of truth for architecture, models, resource envelopes, state machines, and research citations. All code comments and tests citing `PRD §X` reference this document.

---

## Documentation Structure

```
docs/
├── FINAL_SYSTEM_DESIGN.md       # Master Consolidated Final Design (Compiled from 71 Review Gates)
├── PRD.md                       # Canonical Master Requirements & Architecture Spec
├── DEV_NOTES.md                 # Engineering Dev Notes, Component Contracts & Feature Audit
├── RESEARCH_PAPER.md            # Formal Research Paper: Bounded Multi-Turn Agentic UI Design
├── architecture/                # Cross-cutting system design & research justification
│   ├── RESEARCH_AND_CITATIONS.md # Every architectural choice grounded in checked citations
│   ├── DIRECTORY_LAYOUT.md       # Package architecture and boundary rules
│   ├── SYNC_DESIGN.md            # Data-forge ↔ Training sync contract & manifest schemas
│   └── TRAINING_INFERENCE_SYNC_DESIGN.md # Multi-turn context, prompt grounding & weights handoff
│   └── INFERENCE_PRODUCTION_STACK.md # Rust/Python/Tauri runtime boundaries and fail-closed gates
│   └── EVALUATION_PROTOCOL.md     # Unified multi-metric receipt and validation standards
├── data-forge/                  # Data engineering & pipeline architecture
│   ├── README.md                # Master index, architectural orientation, and overview
│   ├── ARCHITECTURE.md          # 22-stage chunk-based processing pipeline & methodology
│   ├── DATA_SOURCES.md          # Public datasets: licenses, schemas, and 5-pillar portfolio
│   └── LITERATURE_SURVEY.md     # Academic literature survey & credits to prior research
├── training/                    # Model training documentation & runbooks
│   ├── README.md                # Training documentation index and current status
│   ├── ARCHITECTURE.md          # Current training design, gates, and bibliography
│   └── TRAINING_RUNBOOK.md      # Readiness checks and current execution guide
├── inference/                   # Serving, orchestration & client integration
│   └── README.md                # Swap state machine, VRAM ledgers & API reference
├── research/                    # Master multi-volume compendiums
│   ├── PROGRESS_AND_ARCHITECTURE_VOLUME_1.md # Architecture evolution & ablation studies
│   ├── PROGRESS_AND_DATAFORGE_VOLUME_2.md    # 22-stage data-forge & multimodal portfolio
│   └── PROGRESS_AND_SYSTEMS_VOLUME_3.md      # Systems serving, OSDI proofs & verification
└── review/                      # Formal verification audit log
    └── README.md                # Consolidated Research Archive and 71-gate phase mapping
```

---

## Key Guides & Quick Links

- **Master Final System Design**: See [FINAL_SYSTEM_DESIGN.md](FINAL_SYSTEM_DESIGN.md) for the consolidated, authoritative specification compiled from all 71 forensic review gates, systems serving literature, and peer-reviewed algorithms.
- **Developer Notes & Feature Retention Audit**: See [DEV_NOTES.md](DEV_NOTES.md) for engineering history, component contracts, memory ledgers, and feature audit.
- **Scientific Research Paper Specification**: See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for formal problem formulation, mathematical models, empirical evaluation, and ablation studies.
- **Current training status and model trainability**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md).
- **Single-GPU Swap Orchestrator**: See [inference/README.md](inference/README.md) and [PRD.md §7](PRD.md#7-inference-orchestration).
- **Production Docker & Dual-Venv Isolation**: See [inference/README.md §5](inference/README.md#5-production-docker-containerization--dual-venv-isolation) and [PRD.md §7.7](PRD.md#77-production-docker-containerization-dual-venv-isolation).
- **Hardware Preflight & Fail-Fast Enforcement**: See [inference/README.md §6](inference/README.md#6-fail-fast-hardware--deployment-preflight-verification) and [PRD.md §7.8](PRD.md#78-hardware--deployment-preflight-verification-fail-fast).
- **Structure/Layout baseline and future learned ablations**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md#2-the-authoritative-training-pipeline).
- **Deferred preference alignment**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md#36-flow-matching-and-preference-alignment).
- **Master Review Archive & 71-Gate Mapping**: See [review/README.md](review/README.md).
- **Volume I: Tier Architecture Evolution & Ablation Studies**: See [research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md](research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md).
- **Volume II: Data-Forge 22-Stage Pipeline & 2.68TB Portfolio**: See [research/PROGRESS_AND_DATAFORGE_VOLUME_2.md](research/PROGRESS_AND_DATAFORGE_VOLUME_2.md).
- **Volume III: Systems Serving, Latency Foundations & Verification**: See [research/PROGRESS_AND_SYSTEMS_VOLUME_3.md](research/PROGRESS_AND_SYSTEMS_VOLUME_3.md).
- **Data-Forge Academic Literature Survey & Prior Research Attribution**: See [data-forge/LITERATURE_SURVEY.md](data-forge/LITERATURE_SURVEY.md).
- **Canonical Peer-Reviewed Citations Matrix (Zero arXiv)**: See [citations.md](citations.md).
- **Hardware Runbook (RTX A6000 48GB)**: See [training/TRAINING_RUNBOOK.md](training/TRAINING_RUNBOOK.md).


Each package's own `README.md` is the source of truth for that package —
this `docs/` tree holds cross-cutting design docs that don't belong to
any one package, plus data-forge's docs (which were migrated here
verbatim from its own former `docs/` directory, unchanged, since
data-forge itself wasn't split).
