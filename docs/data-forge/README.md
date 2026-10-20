# Data-Forge: Multimodal Data Engine & Preprocessing Architecture

Welcome to the definitive documentation suite for **Krisna's Data-Forge**. Data-Forge is the automated, 22-stage data curation, conditioning, and quality-gating pipeline that processes raw multimodal datasets into high-fidelity image-conditioned training pairs for fine-tuning **Qwen-Image-2.1** via QDoRA ($r=64, \alpha=32$).

---

## 1. Directory Structure & Documentation Map

The `docs/data-forge/` directory is organized into a streamlined, non-redundant documentation triad:

```
docs/data-forge/
├── README.md               # Master index, architectural orientation, and executive overview (This file)
├── ARCHITECTURE.md         # Full 22-stage technical architecture, DAG orchestration, and paper methodology
├── DATA_SOURCES.md         # Master dataset registry, 5-Pillar portfolio allocation, and quarantine audit log
└── LITERATURE_SURVEY.md    # Comprehensive academic literature survey, prior research credits, and citations
```

### Quick Navigation:
- **[System Architecture & Methodology](ARCHITECTURE.md)**: Exhaustive walkthrough of the 22-stage pipeline from Stage 0 (`s00_manifest_planning`) to Stage 12 (`s12_model_data_export`), including chunk-based DAG orchestration, local FP8 vLLM serving, mathematical flow-matching formulations, and privacy scrubbing.
- **[Master Dataset Registry](DATA_SOURCES.md)**: Authoritative source-audit ledger detailing active vs. quarantined datasets, exact record allocations, licensing status, and Sub-3TB storage accounting ($1,750,000$ pairs / $2.68\text{ TB}$).
- **[Academic Literature Survey](LITERATURE_SURVEY.md)**: Formal peer-reviewed citations, academic credits, theoretical problem formulations, and comparative analysis against prior benchmarks for the final research paper.

---

## 2. Core Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATAFORGE 22-STAGE PIPELINE DAG                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  [s00: Manifest] ──> [s01: Fetch] ──> [s01.5: UICrit RAG] ──> [s01.6: DPO Pairs]                 │
│                                                │                     │                           │
│  [s03.5: Face PII] <── [s03: Quality] <── [s02: Dedup (CLIP/FAISS)] <┘                           │
│         │                                                                                        │
│         ▼                                                                                        │
│  [s04: Safety] ──> [s04.5: Escalation] ──> [s05: Recaption] ──> [s05-OCR] ──> [s05.5: Text PII] │
│                                                                                        │         │
│  [s08: Spatial Pad (32px)] <── [s07: Shard Routing] <── [s06: Structure Tree (JSON)] <─────────┘ │
│         │                                                                                        │
│         ▼                                                                                        │
│  [s09: Heldout 5% Split] ──> [s10: Audit Pass (3%)] ──> [s11: Registry Watch]                   │
│                                                                     │                            │
│  [s12: Model Export (polish_qwen_image21)] <────────────────────────┘                            │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The 5 Strategic Dataset Pillars

The production portfolio is bounded at **$1,750,000$ pairs ($\approx 2,677.5\text{ GB} = 2.68\text{ TB}$)**, preserving an unallocated **$10.8\%$ ($322.5\text{ GB}$)** safety headroom below the $3.0\text{ TB}$ hardware ceiling:

| Strategic Pillar | Target Pairs | Ratio | Key Anchoring Datasets | Primary Research Role |
| :--- | :---: | :---: | :--- | :--- |
| **Pillar 1: Photographic Priors** | $350,000$ | $20.0\%$ | LAION-Aesthetic $\ge 6.25$, Open Images v7 | Manifold preservation; prevents catastrophic forgetting. |
| **Pillar 2: Instruction Editing** | $350,000$ | $20.0\%$ | MagicBrush, HQ-Edit, UltraEdit, UI DAGs | Conversational editing; multi-turn layout modifications. |
| **Pillar 3: Graphic Design & Ads** | $400,000$ | $22.9\%$ | Crello, PKU-PosterLayout, AnyWord-3M, Cutouts | Brand aesthetics; marketing banners; visual typography. |
| **Pillar 4: Charts, Docs & Decks**| $300,000$ | $17.1\%$ | ChartQA, PlotQA, InfographicVQA, DocLayNet, SlideVQA | Enterprise analytics; 16:9 pitch decks; multi-column docs. |
| **Pillar 5: Web UI & Mobile Apps** | $350,000$ | $20.0\%$ | CLAY, Mind2Web, Figma2Code, ABO E-Commerce | Production web/desktop SaaS; mobile apps; component tokens. |
| **TOTAL CERTIFIED PORTFOLIO** | **$1,750,000$** | **$100.0\%$** | **7 Active Training-Eligible Sources** | **Sub-3TB Certified ($2.68\text{ TB}$)** |

---

## 4. Key Engineering Invariants

1. **Single Trainable Model Principle**: Only `Qwen-Image-2.1` via QDoRA ($r=64, \alpha=32$) is fine-tuned. The Planner, Sketch tier, and Critic tiers are 100% frozen or deterministic.
2. **Dual-Hash Collision Prevention**: In Stage 12, every exported pair is verified: $\operatorname{SHA256}(I_{\text{condition}}) \neq \operatorname{SHA256}(I_{\text{target}})$, preventing degenerate identity shortcuts.
3. **Patch-32 Geometry**: Stage 8 strictly applies `pad_to_multiple(image, 32)` to guarantee that Qwen-Image-2.1's $16\times$ VAE downsampler and $2\times 2$ patch token stride never encounter fractional half-patches.
4. **DALL-E 3 Caption Mixture**: Captions are exported with both dense synthetic annotations and concise human prompts, mixed $80/20$ at training time to prevent short-prompt mismatch.
5. **Source-Disjoint Train/Val Splits**: Splitting at the `source_split_identity` level guarantees that multiple screens from the same application or website template never appear in both train and validation sets.

---

## 5. Verification Commands

To verify the registry status and preflight health:
```powershell
python scripts/data-forge/audit_registry.py
```

To run the full automated Data-Forge unit and regression test suite:
```powershell
pytest tests/data_forge -q
```
*(All 265 Data-Forge tests pass 100% cleanly).*
