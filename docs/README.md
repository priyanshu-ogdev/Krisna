# Documentation Index

> [!IMPORTANT]
> **Canonical Product & Research Specification**: [PRD.md](PRD.md)
> The authoritative single source of truth for architecture, models, resource envelopes, state machines, and research citations. All code comments and tests citing `PRD §X` reference this document.

---

## Documentation Structure

```
docs/
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
│   ├── ARCHITECTURE.md          # Multi-stage chunk-based processing pipeline
│   ├── DATA_SOURCES.md          # Public datasets: licenses, schemas, and fetch modes
│   └── DATA_COMPLETENESS.md     # Audit of data transformations, PII, and safety filters
├── training/                    # Model training documentation & runbooks
│   ├── README.md                # Training documentation index and current status
│   ├── ARCHITECTURE.md          # Current training design, gates, and bibliography
│   └── TRAINING_RUNBOOK.md      # Readiness checks and current execution guide
├── inference/                   # Serving, orchestration & client integration
│   └── README.md                # Swap state machine, VRAM ledgers & API reference
└── review/                      # Formal verification audit log
    ├── README.md                # Audit index and phase roadmap
    ├── 07_consolidated_citations.md # Paper-ready academic bibliography
    └── 28_docker_isolation_and_hardware_preflight.md # Containerization & hardware verification
```

---

## Key Guides & Quick Links

- **Developer Notes & Feature Retention Audit**: See [DEV_NOTES.md](DEV_NOTES.md) for engineering history, component contracts, memory ledgers, and feature audit.
- **Scientific Research Paper Specification**: See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for formal problem formulation, mathematical models, empirical evaluation, and ablation studies.
- **Current training status and model trainability**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md).
- **Single-GPU Swap Orchestrator**: See [inference/README.md](inference/README.md) and [PRD.md §7](PRD.md#7-inference-orchestration).
- **Production Docker & Dual-Venv Isolation**: See [inference/README.md §5](inference/README.md#5-production-docker-containerization--dual-venv-isolation) and [PRD.md §7.7](PRD.md#77-production-docker-containerization-dual-venv-isolation).
- **Hardware Preflight & Fail-Fast Enforcement**: See [inference/README.md §6](inference/README.md#6-fail-fast-hardware--deployment-preflight-verification) and [PRD.md §7.8](PRD.md#78-hardware--deployment-preflight-verification-fail-fast).
- **Structure/Layout baseline and future learned ablations**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md#2-the-authoritative-training-pipeline).
- **Deferred preference alignment**: See [training/ARCHITECTURE.md](training/ARCHITECTURE.md#36-flow-matching-and-preference-alignment).
- **Hardware Runbook (RTX A6000 48GB)**: See [training/TRAINING_RUNBOOK.md](training/TRAINING_RUNBOOK.md).


Each package's own `README.md` is the source of truth for that package —
this `docs/` tree holds cross-cutting design docs that don't belong to
any one package, plus data-forge's docs (which were migrated here
verbatim from its own former `docs/` directory, unchanged, since
data-forge itself wasn't split).
