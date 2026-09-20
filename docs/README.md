# Krisna Documentation Hub

> [!IMPORTANT]
> **Canonical Product & Research Specification**: [PRD.md](file:///d:/Krisna/docs/PRD.md)
> The authoritative single source of truth for architecture, models, resource envelopes, state machines, and research citations. All code comments and tests citing `PRD §X` reference this document.

---

## Documentation Structure

```
docs/
├── PRD.md                       # Canonical Master Requirements & Architecture Spec
├── architecture/                # Cross-cutting system design & research justification
│   ├── RESEARCH_AND_CITATIONS.md # Every architectural choice grounded in checked citations
│   ├── DIRECTORY_LAYOUT.md       # Package architecture and boundary rules
│   ├── SYNC_DESIGN.md            # Data-forge ↔ Training sync contract & manifest schemas
│   └── TRAINING_INFERENCE_SYNC_DESIGN.md # Multi-turn context, prompt grounding & weights handoff
├── data-forge/                  # Data engineering & pipeline architecture
│   ├── ARCHITECTURE.md          # Multi-stage chunk-based processing pipeline
│   ├── DATA_SOURCES.md          # 9 public datasets: licenses, schemas, and fetch modes
│   └── DATA_COMPLETENESS.md     # Audit of data transformations, PII, and safety filters
├── training/                    # Model training documentation & runbooks
│   ├── README.md                # Quick reference, loss mathematics & commands
│   └── TRAINING_RUNBOOK.md      # Step-by-step RTX A6000 48GB execution guide
├── inference/                   # Serving, orchestration & client integration
│   └── README.md                # Swap state machine, VRAM ledgers & API reference
└── review/                      # Formal 28-phase verification audit log
    ├── README.md                # Audit index and phase roadmap
    ├── 07_consolidated_citations.md # Paper-ready academic bibliography
    └── 28_docker_isolation_and_hardware_preflight.md # Containerization & hardware verification
```

---

## Key Guides & Quick Links

- **Why only two models are trained**: See [RESEARCH_AND_CITATIONS.md](file:///d:/Krisna/docs/architecture/RESEARCH_AND_CITATIONS.md) §1 and [PRD.md §6](file:///d:/Krisna/docs/PRD.md#6-model-stack-the-no-rlhf-loop-revision).
- **Single-GPU Swap Orchestrator**: See [inference/README.md](file:///d:/Krisna/docs/inference/README.md) and [PRD.md §7](file:///d:/Krisna/docs/PRD.md#7-inference-orchestration).
- **Production Docker & Dual-Venv Isolation**: See [inference/README.md §5](file:///d:/Krisna/docs/inference/README.md#5-production-docker-containerization--dual-venv-isolation) and [PRD.md §7.7](file:///d:/Krisna/docs/PRD.md#77-production-docker-containerization-dual-venv-isolation).
- **Hardware Preflight & Fail-Fast Enforcement**: See [inference/README.md §6](file:///d:/Krisna/docs/inference/README.md#6-fail-fast-hardware--deployment-preflight-verification) and [PRD.md §7.8](file:///d:/Krisna/docs/PRD.md#78-hardware--deployment-preflight-verification-fail-fast).
- **Progressive Sketch Training (MaskGIT)**: See [training/README.md](file:///d:/Krisna/docs/training/README.md) and [PRD.md §6.1](file:///d:/Krisna/docs/PRD.md#61-progressive-resolution-training-sketch-tier).
- **Diffusion-DPO for Flow-Matching**: See [RESEARCH_AND_CITATIONS.md](file:///d:/Krisna/docs/architecture/RESEARCH_AND_CITATIONS.md) §4.5 and [PRD.md §8.5](file:///d:/Krisna/docs/PRD.md#85-preference-pairs-a-separate-stream).
- **Hardware Runbook (RTX A6000 48GB)**: See [training/TRAINING_RUNBOOK.md](file:///d:/Krisna/docs/training/TRAINING_RUNBOOK.md).

