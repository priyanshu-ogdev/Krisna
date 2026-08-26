# The Master Dataset Registry (Source Ledger)

This is the definitive list of data sources the Data-Forge is configured to ingest. This satisfies the EU AI Act / data provenance requirements.

| Dataset Name | Source / Repo ID | Domain | License Status | Pipeline Usage |
| :--- | :--- | :--- | :--- | :--- |
| **PD12M** | `pixparse/pd12m` (HuggingFace) | General Visual | CC0 / Public Domain | Primary backbone for continuous renderer (Z-Image/Qwen) aesthetics and composition. |
| **CC12M** | `google-research-datasets/conceptual-12m` | General Visual | Unverified / Scraped | Supplementary diversity. Strictly filtered by Safety & Aesthetic gates. |
| **RICO** | `interactionmining.org/rico` | Mobile UI | Academic / Unverified | Base UI layout distribution. Routed to `excluded_pending_review` by License Agent unless verified. |
| **CLAY** | `google-research-datasets/clay` (GitHub) | Mobile UI | Academic / Archived | Denoised RICO. Preferred over raw RICO for structural JSON training. |
| **Enrico** | `enrico.design` | Mobile UI | Academic | High-quality UI subset. Used primarily for **Held-Out Eval** and Verifier calibration. |
| **WebUI** | `webui-dataset.github.io` | Web/Desktop UI | Academic | Extends UI coverage beyond mobile to complex web dashboards and landing pages. |
| **Screen2Words** | `google-research-datasets/screen2words` | UI Captions | Apache 2.0 | Dense caption training pairs for the Planner and Recaption Agent. |
| **UICrit** | `uicrit.com` / Associated HF Repo | UI Critique | Academic | **Critical:** Seeds the DPO preference pairs and trains the Tier-1 Audit Agent rubric. |
| **Synthetic Self-Play** | *Generated Locally by Krisna* | Agentic Edits | Internal | Iterative design trajectories (Bad Design -> Critique -> Fix) generated post-M1. |
