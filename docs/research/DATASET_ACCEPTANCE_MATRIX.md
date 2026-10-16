# Dataset acceptance matrix

Status: candidate review checkpoint, 2026-10-14

## Negative audit outcome (2026-10-14)

The registry was re-audited against the active objective: a real, UI-domain,
deterministic wireframe/layout-condition to polished screenshot target corpus
for Qwen-Image-2.1. The default production registry now has **five enabled
sources**: Figma2Code as the only enabled UI design candidate, plus UI-Vision,
ScreenSpot, TASTE, and PartiPrompts as isolated evaluation sources. It has
**zero training-eligible sources** and **zero preference-pair sources**.

The following sources are disabled in `configs/datasets.yaml` and remain only
as auditable quarantine records: PD12M, Pick-a-Pic v2, HPDv2, SeeClick,
Android Control, Multimodal Mind2Web, ShowUI Desktop, Wave-UI, WebSight,
GameLabel-10K, UICrit, MobileViews, Aria-UI, WebUI, and AtomBlock-WebUI.
Disabling is deliberate rather than deleting provenance: it prevents accidental
fetching while preserving the reason for rejection.

Negative-audit reasons include general-domain mismatch, synthetic/LLM-generated
content, third-party app or web asset rights uncertainty, preference data being
outside supervised paired training, unstable or unpinned mirrors, and source
schemas that cannot produce a deterministic condition/target pair. In
particular, WebSight and AtomBlock-WebUI are not acceptable as real-data
substitutes: their cards explicitly describe synthetic HTML/rendering and, for
AtomBlock-WebUI, injected CC3M imagery and CC BY-NC-SA restrictions.

The online evidence used for this revision includes the UI-Vision card
(MIT, 1,464-row desktop evaluation benchmark), ScreenSpot card (Apache-2.0,
1,272-row test benchmark), MobileViews card (MIT metadata claim but a reported
475 GB repository of third-party Google Play captures), Multimodal Mind2Web
card (OpenRAIL and an explicit warning that some training screenshots may be
improperly rendered), ShowUI Desktop's 7,496-row annotation corpus, Figma2Code's
2,817-row screenshot/metadata benchmark, WebSight's synthetic tag and card,
AtomBlock-WebUI's synthetic-generation and inherited-source disclosure, and
UICrit's paper (3,059 critiques over 983 mobile UIs). These are evidence for
classification, not a substitute for a pinned release receipt or legal review.

This is a governance record, not a claim that the listed sources have been
downloaded, legally cleared, or realized locally. The checked-in Data-Forge
registry is intentionally a **candidate catalog**. A source becomes eligible
for a model-training export only after all of the following are recorded:

1. an immutable upstream revision (full commit SHA);
2. a verified license decision for the actual image/annotation assets;
3. a provenance URL and source-revision receipt on every exported record;
4. a reviewed sample with no unsupported synthetic or derivative-content claim;
5. a target-specific training assignment; and
6. source-identity-disjoint train/held-out splitting.

## Current verdicts

The pairing/annotation descriptions in this table are registry descriptions,
not independent legal or count verification. The **Current verdict** and
**Blocking evidence** columns control acceptance; a source is not approved
merely because its description sounds suitable.

| Source group | Intended use | Pairing / annotation value | Current verdict | Blocking evidence |
|---|---|---|---|---|
| ScreenSpot, UI-Vision, SeeClick, ShowUI Desktop, Wave-UI | UI structure and Planner/VL support | Real UI screenshots and interaction/grounding annotations; not automatically edit pairs | Candidate only | Registry revisions are `main`; license and asset provenance still need a pinned receipt and sample review |
| Android Control, MobileViews, Aria-UI | Mobile/cross-platform coverage | Useful distribution coverage for structure and UI grounding | Candidate only | Same revision/provenance gate; do not infer redistribution rights from a dataset-card summary |
| Figma2Code, WebUI | Structural/UI design research | Figma/code or web UI metadata may support structure analysis, but does not by itself create a condition/target pair | Candidate only | Asset and derivative-rights review; WebUI is annotation-only in the current registry |
| PD12M | General visual polish pretraining research | General images; not a UI pair source | Not approved for current Qwen pair export | General-visual source and unverified license/revision; cannot satisfy the UI-pair contract |
| Pick-a-Pic v2, HPDv2, GameLabel-10K | Preference evaluation/alignment research | Human preference comparisons can support a deferred DPO study, not Qwen condition/target training | Deferred | Separate preference stream; current revisions and license receipts are not pinned |
| UICrit | Planner-side critique retrieval and audit research | Human critique annotations; annotation join, not standalone image training | Manual legal review | NoDerivs/annotation transformation concerns and unpinned Git revision |
| TASTE, PartiPrompts | Held-out evaluation | Expert multi-axis ratings or prompt diversity | Evaluation only | Must remain structurally unreachable from training exports |
| WebSight, AtomBlock-WebUI | Synthetic structural research | Synthetic HTML/layout distributions can be useful for experiments | Excluded from approved training | Synthetic sources are disallowed by the current export policy |

## Evidence labels and capacity profile

The matrix uses two different kinds of evidence deliberately:

| Label | What is verified in this checkout | What it does **not** prove |
|---|---|---|
| **Registry-verified** | A source identifier/provenance URL, explicit synthetic status, fetch shape, and configured cap are recorded in `configs/datasets.yaml`. | That the upstream files still exist, that the advertised count is realized, or that the asset license covers this project's transformed export. |
| **Upstream-verified** | A license or dataset fact is described in `license_notes` with a source URL and the registry records `license_status: verified`. | That a mutable `main`/`master` revision is reproducible, or that every image/annotation asset has the same rights. |
| **Unverified / pending** | `license_status: unverified`, `synthetic_status: unknown`, a missing public repository, or a mutable branch is kept visible as a blocker. | Eligibility for model training. No such source is training-eligible in the checked-in catalog. |

`expected_record_count` and `fetch_config.sample_size` below are **capacity
metadata only**. The first is a registry claim and the second is a configured
cap; they are not post-filter counts and do not mean that data has been
downloaded. The storage class is derived from the configured fetch contract and
the `DatasetSpec` accounting helpers, not from a measured run.

| Capacity profile | Registry entries | Capacity basis | Storage/accounting interpretation | Acceptance boundary |
|---|---|---|---|---|
| Sampled image records | `pd12m`, `figma2code`, `seeclick`, `android_control`, `mind2web`, `wave_ui`, `websight`, `mobileviews`, `aria_ui`, `atomblock_webui` | `min(expected_record_count, fetch_config.sample_size)` | Single-image record budget; configured cap only | Candidate until revision, asset rights, and sample review are complete; synthetic entries remain excluded |
| Uncapped image records | `ui_vision`, `screenspot`, `showui_desktop` | `expected_record_count` when no sample cap is declared | Single-image accounting only where the fetch mode creates image records | A missing cap is not a promise to fetch the full registry claim |
| Sampled preference pairs | `pickapic_v2`, `hpdv2`, `gamelabel_10k` | `min(expected_record_count, fetch_config.sample_size)` | Preference-pair budget, separate from single-image storage | Deferred alignment research; no Qwen condition/target pairs are implied |
| Annotation-only / join | `uicrit`, `webui` | `0` standalone image records in storage projection | Annotation/join metadata is accounted separately or joined to existing records | Must not be promoted to standalone image training without a reviewed join contract |
| Evaluation reference | `taste`, `partiprompts` | `0` training records | Held-out/evaluation storage path, not training-pool capacity | Structurally unreachable from approved training exports |

The profile is intentionally conservative: a source can have a useful
capacity estimate while still being `training_eligible: false`. Promotion
requires replacing the registry claim with a run receipt containing an
immutable source revision, realized post-filter counts, source-disjoint split
counts, and license/provenance evidence. Until then, capacity numbers must not
be used as a training-set size or a license claim.

## Qwen-Image-2.1 pair contract

The Qwen export is intentionally stricter than the general image export. A
record must be an explicitly approved UI source, have a scrubbed target
screenshot, valid structure output, and carry a renderer-produced
`encoding_paths.condition_image`. The condition must be a deterministic
wireframe/layout image and must not be the target screenshot, a JSON control
map, or a Canny edge map. Missing renderer artifacts are written to
`model_data/polish_qwen_image21/exclusions.jsonl`; no pair is fabricated.

The current workspace has no realized Qwen pair corpus. Consequently this
matrix does not promote any source to `training_eligible: true` in the checked-
in registry. Once a source is cleared, update its registry entry with its
actual commit SHA, verified license decision, provenance receipt policy, and
`training_targets: [polish_qwen_image21]`, then run the focused export tests
before any training launch.

## Data-Forge source expansion (2026-10-14)

Two public sources now have explicit, metadata-preserving adapters and can be
combined in one local manifest without losing upstream identity:

| Source | Primary evidence | Adapter | Data retained | Training status |
|---|---|---|---|---|
| Figma2Code | [HF card](https://huggingface.co/datasets/xcodemind/Figma2Code), [ICLR 2026 paper](https://openreview.net/forum?id=CaXZB6bI31) | `figma2code` / `figma2code_local` | `root` screenshot, `filekey`, `node_id`, raw/processed metadata, source split, assets as metadata only | Structure candidate; no Qwen pair until a deterministic renderer is run |
| MobileViews | [HF card](https://huggingface.co/datasets/mllmTeam/MobileViews), [Gao et al. 2024/2025](https://huggingface.co/datasets/mllmTeam/MobileViews) | `mobileviews` / `mobileviews_local` | `image_content`, `json_content`, package/app group when present, parquet shard, source revision | Disabled by default pending third-party app-asset rights review |

Figma2Code's card describes `root`, `filekey`, `node_id`, `raw_metadata`,
`processed_metadata`, and separate `data_test`/`data_rest` parquet splits.
MobileViews' card describes `image_content`/`json_content` parquet pairs,
app-trace archives, and a claimed MIT dataset-card license, while also
describing screenshots captured from third-party Google Play applications.
The adapter therefore records the card's provenance but does not infer that
the MIT metadata claim clears every embedded screenshot or app asset.

### Local-only preprocessing contract

```powershell
data-forge preprocess --dataset figma2code --source-path D:\datasets\Figma2Code --limit 2500
data-forge preprocess --dataset mobileviews --source-path D:\datasets\MobileViews --limit 2500
```

This command intentionally is not a substitute for the governed model
pipeline. It uses no VLM, no CUDA model, no license-page call, and no synthetic
caption/structure generation. It writes canonical images and source metadata,
inserts records with status `fetched`, and writes a receipt under
`DATA_ROOT/audit_reports/`. Records remain ineligible until the ordinary local
PII, safety, quality, structure, split, audit, and export gates run.

The same adapter code is used after a remote Hugging Face snapshot download
and after a local artifact handoff. This makes the two paths combinable and
keeps identity/hash semantics identical. The adapters read parquet in bounded
batches, hash the bytes actually written, preserve `source_record_id`,
`source_group_id`, `source_revision`, `source_metadata`, and
`source_structure`, and never recursively treat Figma bitmap assets as
independent screens.

### Synthetic-processing decision

No synthetic data is required to make these sources usable for the current
research question. Figma2Code and MobileViews already provide screenshots and
source structure; local preprocessing should normalize and validate what is
present, not ask a model to invent missing labels. A deterministic wireframe
renderer may later derive a *condition image* from validated structure, but it
must be a recorded transformation with input/output hashes—not a human-drawn
design and not a future dataset release. If source structure is missing or
cannot be normalized, the record is reported as incomplete and excluded from
the Qwen pair export rather than filled with synthetic analysis.
