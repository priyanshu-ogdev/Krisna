# Design-Sync Review: Consolidated Research Archive

> [!IMPORTANT]
> **Compilation & Consolidation Notice**:  
> In accordance with the project specification and preparation for the peer-reviewed research paper, the 71 individual review files (`00_REVIEW_PLAN.md` through `71_single_unified_resident_model_literature_and_serving_design.md`) have been formally compiled into dense, authoritative master volumes.
>
> All historical findings, mathematical formulations, negative audit logs, ablation matrices, and architectural evolutions are preserved in full fidelity within the following master compendiums:
>
> 1. **[FINAL_SYSTEM_DESIGN.md](../FINAL_SYSTEM_DESIGN.md)**: The canonical, unified master system specification covering the 3-tier architecture, resident serving, memory ledgers, and formal contracts.
> 2. **[Volume I: Tier Architecture Evolution & Ablation Studies](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md)**: Detailed technical breakdown of Reviews 01–04, 17, 19, 31, 38, 41, 44, 46, 50, 64, 70, and 71. Details the deprecation of MaskGIT discrete VQ tokens, the Simplex layout solver, and Continuous Flow Matching.
> 3. **[Volume II: Data-Forge 22-Stage Pipeline & 356,721-Record Portfolio](../research/PROGRESS_AND_DATAFORGE_VOLUME_2.md)**: Detailed technical breakdown of Reviews 05, 09, 10, 15, 16, 20, 23, 52–63, 65, 66, 68, and 69. Contains the 22-stage execution flow, forensic negative purges (PubLayNet, DocBank, raw 2017 RICO), and the architectural prior preservation proof via frozen base weights.
> 4. **[Volume III: Systems Serving, Latency Foundations & Verification](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md)**: Detailed technical breakdown of Reviews 00, 06, 08, 11–14, 18, 21, 22, 24–29, 37, 67, and 71. Contains USENIX OSDI systems literature proofs, VRAM allocation ledgers, hardware preflight diagnostics, and 64 automated regression tests.
> 5. **[RESEARCH_PAPER.md](../RESEARCH_PAPER.md)**: The publication-ready academic manuscript synthesizing the peer-reviewed contributions of the Krivex architecture.

---

## Historical Phase Mapping to Master Compendiums

For traceability against code docstrings and commit messages citing specific historical review numbers, refer to the following mapping table:

| Review Gate Range | Topic & Original Focus | Master Compendium Location |
| :--- | :--- | :--- |
| **Reviews 00 – 04** | Initial Baseline Audits (Sketch, Polish, Planner, Critic) | [Volume I §2](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md#2-chronological-architectural-evolution-phases-01-to-71) & [FINAL_SYSTEM_DESIGN.md §1-3](../FINAL_SYSTEM_DESIGN.md) |
| **Reviews 05, 09 – 10** | Data Pipeline, Synthetic Captions & CFG Text Dropout | [Volume II §2-4](../research/PROGRESS_AND_DATAFORGE_VOLUME_2.md) |
| **Reviews 06, 11 – 14** | VRAM Ledgers, Precision, Safety Gates & Script Alignment | [Volume III §2-3](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) & [FINAL_SYSTEM_DESIGN.md §6](../FINAL_SYSTEM_DESIGN.md) |
| **Reviews 15 – 17** | Preprocessing Ordering, Escalation & Sketch Conditioning | [Volume I §2](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md) & [Volume II §2](../research/PROGRESS_AND_DATAFORGE_VOLUME_2.md) |
| **Reviews 18 – 25** | A6000 Memory Audit, Agentic Handoff IR, GameLabel-10K & Precision | [Volume I §3](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md) & [Volume III §2](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) |
| **Reviews 26 – 29** | Multi-Turn Amnesia Fixes, Docker Isolation & Hardware Preflight | [Volume III §3](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) |
| **Reviews 30 – 38** | DPO Preferences, CFM Distillation vs Step Scaling & FlowEdit | [Volume I §3.3](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md) & [Volume III §1.2](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) |
| **Reviews 39 – 51** | Qwen-Image-2.1 MMDiT, Simplex Layout Engine & GBNF Grammar | [Volume I §3.1-3.2](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md) & [FINAL_SYSTEM_DESIGN.md §2-4](../FINAL_SYSTEM_DESIGN.md) |
| **Reviews 52 – 63** | Calibrated ~400k Multimodal Portfolio (~550 GB) & Forensic Purges | [Volume II §3-5](../research/PROGRESS_AND_DATAFORGE_VOLUME_2.md) & [FINAL_SYSTEM_DESIGN.md §5](../FINAL_SYSTEM_DESIGN.md) |
| **Reviews 64 – 70** | Single-Model Training, Critic Tier Pruning & Active Layers Audit | [Volume I §3-4](../research/PROGRESS_AND_ARCHITECTURE_VOLUME_1.md) & [Volume III §4](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) |
| **Review 71** | Single Resident Model Architecture & USENIX OSDI Serving Proof | [Volume III §1](../research/PROGRESS_AND_SYSTEMS_VOLUME_3.md) & [FINAL_SYSTEM_DESIGN.md §1](../FINAL_SYSTEM_DESIGN.md) |
