# Krisna: Research & Formal Specifications Index

This directory holds the research records, architectural audits, and workflow designs guiding Krisna's development.

## Core Documents

1. **[Formal Research Paper Specification](../RESEARCH_PAPER.md)**:
   - *Title*: "Krisna: Bounded Multi-Turn Agentic UI Design via Structured Layout IR and Unified Diffusion Rendering"
   - Formal mathematical formulation, state space, layout geometry, diffusion conditioning, 4-turn budget convergence, empirical benchmark evaluations (RICO, CLAY, Enrico, Screen2Words, GameLabel-10K, UICrit), and comprehensive ablation studies.

2. **[Developer Notes & Engineering Record](../DEV_NOTES.md)**:
   - System overview, component specifications, feature retention audit, memory ledgers (12GB VRAM / 16GB host RAM), and testing runbooks.

3. **[Agentic Pipeline Architecture Review](agentic_pipeline_architecture_review.md)**:
   - Audit of turn budget policy, inactivity debounce coordinator, session context limits, and finalization gating.

4. **[Phase 5: Agentic Workflow Design](phase5_agentic_workflow_design.md)**:
   - Single-model-per-responsibility rules, Qwen-Image-2.1 DreamBooth LoRA training contracts, adapter manifests, and failure recovery policies.

5. **[Research Citations & Bibliography](../citations.md)**:
   - Comprehensive citations for layout flow matching, grammar-guided generation, hybrid retrieval, and diffusion alignment.
