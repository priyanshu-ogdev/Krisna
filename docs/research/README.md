# Krisna: Research & Formal Specifications Index

This directory holds the research records, architectural audits, and workflow designs guiding Krisna's development.

> **Start with the final audit:** [`FINAL_ARCHITECTURE_VERDICT.md`](FINAL_ARCHITECTURE_VERDICT.md)
> is the binding model-selection, train/freeze, negative-finding, evaluation,
> and publication-readiness decision. The older specifications below retain
> historical reasoning but may contain superseded model names or provisional
> measurements.

## Core Documents

1. **[Formal Research Paper Specification](../RESEARCH_PAPER.md)**:
   - *Title*: "Krisna: Bounded Multi-Turn Agentic UI Design via Structured Layout IR and Unified Diffusion Rendering"
   - Pre-registration/specification with historical result placeholders clearly marked as unvalidated.

2. **[Final Architecture Verdict](FINAL_ARCHITECTURE_VERDICT.md)**:
    - Audited implementation status, exact model/freeze matrix, Qwen Planner–Polish synchronization, gated training plan, negative-point audit, public evaluation protocol, and primary sources.

3. **[Dataset Acceptance Matrix](DATASET_ACCEPTANCE_MATRIX.md)**:
    - Current source-governance verdicts, provenance/license gates, and the strict Qwen condition/target pair contract. This is a candidate review record, not evidence of a realized corpus.

4. **[Data-Forge Local Preprocessing](DATAFORGE_LOCAL_PREPROCESSING.md)**:
    - Dataset evidence, canonical provenance, local-only preprocessing stages, source synchronization, and the no-invented-label decision.

5. **[SOTA Audit — October 2026](SOTA_AUDIT_2026-10.md)**:
    - Current external research scan, actionable upgrades, source links, and
      explicit non-transferability rules for model and sampler results.

6. **[Flow-Matching Upgrade Plan](FLOW_MATCHING_UPGRADE_PLAN.md)**:
    - Publisher-backed flow/editing rationale, upgraded native training and
      inference contracts, receipts, ablations, and hardware gates.

7. **[Research-paper Artifact Map](RESEARCH_ARTIFACT_MAP.md)**:
    - Section-by-section mapping from paper claims to implementation modules,
      required receipts, and publication status.

8. **[Strict Local VLM Preprocessing Architecture](../data-forge/LOCAL_VLM_PREPROCESSING_ARCHITECTURE.md)**:
    - Pinned Qwen3.8-27B local runtime, 48 GiB VRAM/128 GiB RAM allocation, no-fallback policy, commands, and final training-data gates.

9. **[Developer Notes & Engineering Record](../DEV_NOTES.md)**:
   - System overview, component specifications, feature retention audit, memory ledgers (12GB VRAM / 16GB host RAM), and testing runbooks.

10. **[Agentic Pipeline Architecture Review](agentic_pipeline_architecture_review.md)**:
   - Audit of turn budget policy, inactivity debounce coordinator, session context limits, and finalization gating.

11. **[Phase 5: Agentic Workflow Design](phase5_agentic_workflow_design.md)**:
   - Single-model-per-responsibility rules, Qwen-Image-2.1 DreamBooth LoRA training contracts, adapter manifests, and failure recovery policies.

12. **[Research Citations & Bibliography](../citations.md)**:
   - Comprehensive citations for layout flow matching, grammar-guided generation, hybrid retrieval, and diffusion alignment.
