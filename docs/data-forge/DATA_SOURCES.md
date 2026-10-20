# The Master Dataset Registry (Source Ledger)

> [!IMPORTANT]
> **Canonical Literature Survey**: See [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) for the formal theoretical analysis, methodology evaluation, and academic credits for all prior research underpinning Data-Forge.
>
> This document is the source-audit ledger distinguishing active training sources from quarantined catalog entries. A source is not training-approved merely because it appears in this document; the runtime registry and immutable receipt gates (`audit_registry.py`) are authoritative.

## Current Audit State & Sub-3TB Rebalanced Portfolio

The production training portfolio is bounded at **$1,750,000$ pairs ($\approx 2,677.5\text{ GB} = 2.68\text{ TB}$)**, preserving an **$10.8\%$ ($322.5\text{ GB}$)** unallocated safety buffer below the $3.0\text{ TB}$ ceiling. General-purpose rehearsal is anchored at the **$20.0\%$ Pareto sweet spot** ([Volume 2 §4](../research/PROGRESS_AND_DATAFORGE_VOLUME_2.md#4-forensic-audit-of-pillar-4--over-allocation-rebalancing-review-61--62)), while corporate document layout is rebalanced to the **$8.0\%$ SOTA literature ceiling** ([Review Archive](../review/README.md)) to accommodate data visualizations, charts, diagrams, and 16:9 presentation pitch decks.

Run:
```powershell
python scripts/data-forge/audit_registry.py
```
before every fetch. The command is read-only and reports enabled/quarantined sources, effective counts, policy violations, and environment-independent source blockers.

---

## Active 5-Pillar Training Registry

| Pillar & Domain | Dataset Name | Source / Repo ID | License Status | Role & Pipeline Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1: Photo Prior** | **LAION-Aesthetics v2** | `laion/laion-art` / Parquet | CC-BY-4.0 | $250\text{k}$ pairs. Filtered at Aesthetic Score $\ge 6.25$ and DCT blockiness $B \le 0.12$. Dense synthetic recaptioning via Qwen-2.5-VL to anchor natural texture velocity fields. (Schuhmann et al., NeurIPS 2022). |
| **Pillar 1: Photo Prior** | **Open Images v7** | `open-images-v7` (Google) | CC-BY-4.0 / CC-BY-2.0 | $100\text{k}$ pairs. Curated diverse architectural, human, and natural object scenes. Motion-blur filtered ($\sigma_{\nabla}^2 \ge 120.0$). Total Pillar 1: $350,000$ pairs ($20.0\%$). (Kuznetsova et al., IJCV 2020). |
| **Pillar 2: I2I & UI Edit** | **MagicBrush** | `osunlp/MagicBrush` | CC-BY-4.0 (confirmed) | $10,500$ pairs. Ground-truth human-annotated image editing triplets $(I_{\text{source}}, \text{instruction}, I_{\text{target}})$ with precision masks. (Zhang et al., NeurIPS 2023 Oral). |
| **Pillar 2: I2I & UI Edit** | **HQ-Edit** | `UCSC-VLAA/HQ-Edit` | CC-BY-NC-4.0 (confirmed) | $197,350$ pairs. High-resolution DALL-E 3 & GPT-4V aligned conversational photo edits. Filtered via directional CLIP delta $[0.35, 0.82]$. (Huang et al., ICLR 2025). |
| **Pillar 2: I2I & UI Edit** | **UltraEdit (Filtered)** | `UltraEdit/UltraEdit` | CC-BY-4.0 (confirmed) | $112,150$ pairs. Fine-grained localized and global photo editing. Bottom 150k pairs pruned via Laplacian seam boundary continuity ($\Delta \nabla \le 0.05$). (Zhang et al., ECCV 2024). |
| **Pillar 2: I2I & UI Edit** | **UI DAG Mutations** | Synthetic (Internal Krisna) | Internal | $30,000$ pairs. Multi-turn UI layout modification pairs generated via FlowEdit differential masking, teaching deterministic lock preservation. Total Pillar 2: $350,000$ pairs ($20.0\%$). |
| **Pillar 3: Graphic Design** | **Crello (Filtered)** | `cyberagent/crello` | CDLA-Permissive-2.0 | $240,000$ pairs. Commercial marketing banners, social templates, and $>60,000$ isolated transparent PNG layer assets providing ground-truth alpha channels. Regex-purged of *"Lorem Ipsum"*. (Yamaguchi, ACM MM 2021). |
| **Pillar 3: Graphic Design** | **PosterLayout** | `creative-graphic-design/PKU-PosterLayout` | CC-BY-SA-4.0 | $9,974$ pairs. Content-aware visual-textual presentation layout pairs for e-commerce advertising posters. (Hsu et al., IEEE TMM 2023). |
| **Pillar 3: Graphic Design** | **AnyWord-3M (Digital)** | `stzhao/AnyWord-3M` | Apache-2.0 | $90,000$ pairs. Curated digital packaging, posters, and banners with word-level bounding polygon OCR. Excised outdoor street-scene OCR. (Tuo et al., ICLR 2024 Spotlight). |
| **Pillar 3: Graphic Design** | **Alpha Cutout Assets** | Curated Transparent Vectors | Permissive / Apache-2.0 | $60,026$ pairs. High-resolution isolated graphic cutouts, badges, and typography containers with verified binary alpha transparency. Total Pillar 3: $400,000$ pairs ($22.9\%$). |
| **Pillar 4: Charts & Info** | **ChartQA** | `Salesforce/ChartQA` | CC-BY-4.0 | $50,000$ pairs. High-precision bar charts, line graphs, pie charts, and scatter plots paired with underlying tabular data. (Masry et al., ACL 2022). |
| **Pillar 4: Charts & Info** | **PlotQA** | Algorithmic WACV 2020 | MIT / Permissive | $40,000$ pairs. Algorithmic scientific multi-axis plots over World Bank and open government data with verified coordinate bounds. (Methani et al., WACV 2020). |
| **Pillar 4: Charts & Info** | **InfographicVQA** | CVC / IIIT-H WACV 2022 | CC-BY-NC-4.0 | $30,000$ pairs. Multi-tier visual narrative infographics with hierarchical callouts and theme iconography. (Mathew et al., WACV 2022). |
| **Pillar 4: Charts & Info** | **AI2D & Vega-Lite** | `allenai/ai2d` & Declarative | Apache-2.0 / BSD-3 | $40,000$ pairs ($20\text{k}$ AI2D flowcharts/process diagrams $+ 20\text{k}$ Vega-Lite/Plotly modern SaaS dashboard widgets). (Kembhavi et al., ECCV 2016; Santhanam et al., CHI 2023). Track 4A: $160,000$ pairs ($9.1\%$). |
| **Pillar 4: Docs & Slides** | **DocLayNet (Unique)** | `docling-project/DocLayNet` | CDLA-Permissive-1.0 | $80,863$ unique human-annotated pages across 6 enterprise categories. Synthetic duplication eliminated. (Pfitzmann et al., ACM SIGKDD 2022). |
| **Pillar 4: Docs & Slides** | **Condé Nast Magazine** | `xtqiao.com/MagImage.zip` | Academic / Fair Use | $3,919$ verified publication spreads (*Vogue*, *BRIDES*) with multi-column editorial typography. (Zheng et al., ACM SIGGRAPH 2019). |
| **Pillar 4: Docs & Slides** | **SlideVQA & Decks** | `SlideVQA` / Curated Decks | CC-BY-4.0 / Permissive | $25,000$ pairs. Verified 16:9 widescreen landscape presentation slides and business pitch decks. Supplemented by $30,218$ annual reports. Track 4B: $140,000$ pairs ($8.0\%$). Total Pillar 4: $300,000$ pairs ($17.1\%$). (Tanaka et al., AAAI 2023). |
| **Pillar 5: Web & UI** | **CLAY** | `google-research-datasets/clay` | CC-BY-4.0 | $59,000$ pairs. Denoised, vectorized mobile UI screen hierarchies. Completely replaces noisy raw 2017 RICO screens. (Li et al., ACM CHI 2022). |
| **Pillar 5: Web & UI** | **Mind2Web (Desktop)** | `osunlp/Multimodal-Mind2Web` | OpenRAIL | $120,000$ pairs. Desktop web application viewports (1920x1080) processed with viewport cropping (`viewport_crop: true`, `max_aspect_ratio: 2.0`). (Deng et al., NeurIPS 2023). |
| **Pillar 5: Web & UI** | **Design2Code** | `google/Design2Code` | MIT | $5,000$ pairs. Complex multi-column responsive SaaS landing pages and web applications. (Si et al., ACL 2025). |
| **Pillar 5: Web & UI** | **Figma2Code Tokens** | `xcodemind/Figma2Code` | MIT | $30,000$ pairs. Isolated community Figma design system tokens (buttons, modals, search bars, navbars, cards) with transparent canvas alpha. (Gui et al., ICLR 2026). |
| **Pillar 5: Web & UI** | **ABO (Product Cards)** | `wardana/amazon-berkeley-objects`| CC-BY-NC-4.0 | $21,239$ pairs. Commercial e-commerce product cards with ratings, prices, and CTA buttons (isolated CAD voids purged). (Collins et al., CVPR 2022). |
| **Pillar 5: Web & UI** | **Enrico & SaaS Screens**| `luileito/enrico` & Curated | Academic / MIT | $119,761$ pairs ($1,460$ Enrico design motifs $+ 118,301$ modern SaaS/web screens). Total Pillar 5: $350,000$ pairs ($20.0\%$). (Leiva et al., ACM MobileHCI 2020). |

---

## Quarantined & Retired Sources (Negative Audit Log)

| Dataset | Previous Status | Quarantine Reason |
| :--- | :--- | :--- |
| **PubLayNet** | Candidate | **Quarantined**: Severe monochrome LaTeX academic bias. Induces drab aesthetic collapse in generative diffusion backbones. |
| **DocBank** | Candidate | **Quarantined**: Monochrome arXiv paper distribution. Irrelevant to visual UI and modern graphic design. |
| **Raw RICO** | Candidate | **Quarantined**: $65\%$ intra-corpus duplicate screens, broken bounding boxes, and obsolete 2017 Android Marshmallow styling. Fully replaced by CLAY. |
| **WebSight & AtomBlock** | Candidate | **Quarantined**: Synthetic LLM-generated HTML/CSS with severe flex-wrap bugs, overlapping divs, and zero human aesthetic ground truth. |
| **PD12M & CC12M** | Candidate | **Quarantined**: Unfiltered general web crawl. Succeeded by curated LAION $\ge 6.25$ and Open Images v7 with explicit prior preservation conditioning. |

### Vetted but not yet integrated

| Candidate | Repo | Why not wired in yet |
|---|---|---|
| **GameLabel-10K** | `Jonathan-Zhou/GameLabel-10k` (confirmed real, Apache 2.0, arXiv:2409.19830) | **INTEGRATED.** Real, general-domain (not UI-specific) image-preference pairs (~10K labels, 7K prompts) collected via mobile-game crowdsourcing, already demonstrated improving Flux-Schnell via DPO/LoRA fine-tuning. Column schema now confirmed directly against the live dataset (HF's own dataset-viewer output, not guessed): `prompt`, `img0_votes`/`img1_votes` (int 0-5, a vote COUNT from up to 5 players — not a fixed 0/1 label), `img0_encoding`/`img1_encoding` (base64 JPEG wrapped in a stray Python `b'...'` bytes-repr string, not the standard HF `{"bytes": ...}` image-feature dict). Neither `preference_pair` nor `hpdv2_ranked_list` fit this shape, so a dedicated `download_mode: "gamelabel_csv"` adapter was added (`fetcher.py::_fetch_gamelabel_csv`/`_decode_gamelabel_image`), reading via the `refs/convert/parquet` revision Hugging Face auto-generates for this repo rather than streaming the raw 2.26GB `data.csv` directly, deriving a win/lose label from vote counts (dropping ties, including 0-0), and preserving the vote margin per pair for potential downstream weighting. See `datasets.yaml`'s `gamelabel_10k` entry and `tests/data_forge/test_fetch_gamelabel.py` (tie-dropping, malformed-row handling, and label/margin correctness all verified against real pandas/PIL, not just asserted). Supplements Stage-1 (general) DPO, does not close the Stage-2 (UI-domain) gap DesignSense-10k/DesignPref are meant for.
| **Synthetic Self-Play** | *Generated Locally by Krisna* | Agentic Edits | Internal | Iterative design trajectories (Bad Design -> Critique -> Fix) generated post-M1. |
| **Commissioned / rights-cleared gap-filling** | *Not yet sourced* | Domain gaps neither PD12M/CC12M nor the UI datasets cover (PRD §8.1) | N/A | **Intentionally out of scope for data-forge.** This is a manual sourcing/licensing process (commissioning or licensing specific imagery), not an automatable fetch — there is deliberately no `datasets.yaml` entry or fetch path for it. Noted here explicitly so its absence reads as a documented scope boundary, not an overlooked source. |

## Domain Tagging

Records are tagged `ui_first` or `general_design` (`data-forge/src/data_forge/data/domain_tagger.py`) via exact `source_dataset` match against the set above, falling back to a structural heuristic (≥2 overlapping UI element types from Stage 6's extraction) only for unrecognized sources. `rico_core`, `rico_semantic`, `clay`, `enrico`, `webui`, `screen2words`, `screenspot`, `seeclick`, `android_control`, `mind2web`, and `uicrit` are all explicitly `ui_first` sources — all eleven must stay in that source list, or records from any missing one silently fall through to the structure-based heuristic instead of being reliably tagged, undermining the `ui_first_ratio` enforcement in `s07_routing.py` before it even runs.

## Known Limitation: `ui_first_ratio` Is Enforced Per-Chunk, Not Corpus-Wide

`ShardRouter.route()` (`data-forge/src/data_forge/data/shard_router.py`) enforces the configured `ui_first_ratio` (default 0.70) against whichever chunk of records is passed into a single call — it correctly caps the majority domain down and accepts a smaller total routed set when the minority domain can't supply enough to hit the ratio exactly (verified by `tests/test_shard_router.py::test_overflow_excluded`). What it does *not* do is enforce the ratio cumulatively across chunks against a fixed, corpus-wide training-pool size cap — each chunk is ratio-balanced independently. If different chunks have meaningfully different `ui_first`:`general_design` compositions, the corpus-wide realized ratio can drift from the configured target even though every individual chunk is internally correct. Addressing this would require a persistent, cross-chunk counter (e.g. tracked in the manifest) rather than the current fresh-per-call `total`; left as an open item for a future revision rather than expanded in scope here.
