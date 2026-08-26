# Data-Forge — Zero-Touch Data Pipeline (v13)

> The automated data-curation pipeline for **Krisna**, an agentic UI/design system
> (planner → sparse sketch tier → diffusion polish tier). Implements the v13 spec:
> every verification, judgment, and spot-check step that used to require a human
> is now either deterministic logic or a call to a resident reasoning/VLM model,
> logged with a citation or rationale.

## Status (as of this handoff)

| Check | Result |
|---|---|
| Syntax (all `.py`, AST-parsed) | ✅ 0 errors |
| Module imports (all 53 modules) | ✅ 0 real errors — only expected `ModuleNotFoundError` for GPU-only deps (vllm/torch/faiss-gpu/mediapipe) not installed on a non-GPU box |
| Unit tests (`pytest -m "not integration"`) | ✅ 38/38 passing |
| Lint (`ruff check .`) | ⚠️ 57 style-only findings (line length, import order, blind-`Exception` in tests) — no undefined names, no logic errors |
| Packaging (`pyproject.toml` build backend) | ✅ fixed — was pointing at a non-existent backend, now `setuptools.build_meta` |
| **Not yet verified** | Real vLLM/AWQ inference, and a live run against RICO/CLAY/WebUI — both require the target GPU box and network access this environment doesn't have |

**Bottom line:** the codebase is structurally sound and internally consistent with the v13 spec and all four mandatory upgrades (PII scrub, Tri-Path encoding, chunk-based model swapping, storage quota enforcement). What's unverified is specifically the parts that *require your hardware* — not the code logic.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Orchestrator (Chunk-Based)             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────┐ │
│  │Stage 0 │→│Stage 1 │→│Stage 2 │→│Stage 3  │→│ 3.5  │ │
│  │Manifest│ │Fetch+  │ │Dedup   │ │Quality  │ │ PII  │ │
│  │Planning│ │License │ │(FAISS) │ │Scoring  │ │Scrub │ │
│  └────────┘ └────────┘ └────────┘ └─────────┘ └──────┘ │
│       ↓                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │Stage 4 │→│  4.5   │→│Stage 5 │→│Stage 6 │→│Stage 7 │ │
│  │Safety  │ │Escalate│ │Recap+  │ │Struct  │ │Routing │ │
│  │Tier-1  │ │Tier-2  │ │OCR     │ │Extract │ │& Shard │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ │
│       ↓                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │Stage 8 │→│Stage 9 │→│Stage 10│ │Stage 11│            │
│  │Tri-Path│ │Heldout │ │Audit   │ │Registry│            │
│  │Encode  │ │Carve   │ │Pass    │ │Watcher │            │
│  └────────┘ └────────┘ └────────┘ └────────┘            │
└──────────────────────────────────────────────────────────┘
```

Each chunk of records (default 10,000) is passed through the applicable stages
with the relevant model loaded once, then unloaded, before the next model loads —
this is what makes the pipeline viable on a single 48GB card (see `orchestrator.py`).

## Requirements

- **OS**: Ubuntu 22.04 / 24.04 LTS (production), Windows (development only)
- **CUDA**: 12.4 (strictly pinned — vLLM and FAISS-GPU are sensitive to minor-version drift)
- **Python**: 3.10 or 3.11 (`pyproject.toml` currently caps at `<3.12`; loosen only after confirming vLLM's current wheel support for 3.12)
- **GPU**: 48GB VRAM (A6000, RTX 6000 Ada, L40S)
- **RAM**: 128GB system RAM recommended (Tier-1/Tier-2 CPU inference lane)
- **Disk**: ≥3TB at `DATA_ROOT`

## Setup

```bash
cd data-forge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

export DATA_ROOT=/data_krisna
export HF_TOKEN=hf_...
export CUDA_VISIBLE_DEVICES=0
```

## Usage

```bash
data-forge run                              # full pipeline
data-forge run --dry-run                    # validate config + storage, no inference
data-forge run --stages 0,1,2                # run specific stages only
data-forge run --resume                      # resume from last checkpoint

data-forge registry check                   # registry watcher (run weekly via cron/systemd timer)

data-forge manifest stats
data-forge manifest query --status excluded_pending_review
```

## Configuration

All configuration lives in `configs/`:
- `pipeline.yaml` — stage toggles, thresholds, paths, chunk size
- `models.yaml` — pinned model versions/revisions, quantization, VRAM budgets
- `datasets.yaml` — source dataset URLs, license status, category weights

Paths in `pipeline.yaml` are relative to `DATA_ROOT`. Model pins in `models.yaml`
are what `data-forge registry check` proposes updates to — actual swaps are a
deliberate, logged step, never automatic.

## Stage reference

| Stage | Name | GPU Model | Purpose |
|---|---|---|---|
| 00 | Manifest Planning | — | Init DB, storage pre-flight check, read latest registry-watcher report |
| 01 | Fetch & License | Tier-1 (inline) | Download shards, structured license-term extraction |
| 02 | Dedup | CLIP / FAISS | Exact-hash + semantic near-duplicate removal |
| 03 | Quality | Tier-1 | Aesthetic + resolution scoring |
| 03.5 | PII Scrub | MediaPipe / regex | Face blur, sensitive-text redaction |
| 04 | Safety | Tier-1 | NSFW / harmful-content classification |
| 04.5 | Escalation | Tier-2 | Second opinion on borderline safety/license records |
| 05 | Recaption + OCR | Tier-1 → OCR | Dense captioning + text-in-image extraction |
| 06 | Structure | Tier-1 | UI component-tree / layout JSON extraction |
| 07 | Routing | — | Domain tagging, ratio enforcement, shard assignment |
| 08 | Tri-Path Encoding | VAEs / VQ tokenizer | Z-Image latents, Qwen-Image latents, MaskGIT VQ tokens, control maps |
| 09 | Heldout Carve | — | Stratified sampling for the eval set |
| 10 | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge rubric scoring on a 2–5% sample, ensemble-disagreement escalation |
| 11 | Registry Watch | — | External cron; polls HF/GitHub for model/dataset updates, writes a report Stage 0 reads |

## Testing

```bash
pytest tests/ -v -m "not integration"        # unit tests, no GPU required — currently 38/38 passing
pytest tests/ -v -m integration               # requires GPU + downloaded models
pytest tests/ --cov=data_forge --cov-report=html
ruff check .                                  # style/lint
```

## Human-in-the-loop, by design (not an oversight)

Two things are deliberately **not** automated:
1. **Final legal sign-off** on records left in `excluded_pending_review` after
   the License Verification Agent's triage, if the project intends to ship or publish.
2. **The UI-first vs. general-design sampling ratio** in Stage 07 — a product-strategy
   call, not a data-quality one.

Everything else that reads as "verify," "check," or "confirm" in the source
planning docs is now deterministic logic or a logged model call.

## Known gaps / next steps

- [ ] Real vLLM/AWQ load-and-infer smoke test against `models.yaml` pins, on target hardware
- [ ] Small dry run (a few hundred RICO/CLAY records) end-to-end, to get real throughput + audit-pass numbers before committing to the full corpus — feeds directly into the research track's M0/M4 milestones
- [ ] Confirm live RICO/CLAY/WebUI fetch logic still matches current source HTML/API shape (fetcher was written against the sources' shape at spec time, not verified live)
- [ ] Re-evaluate the `<3.12` Python pin once vLLM 3.12 wheel support is confirmed
- [ ] Optional: clear the remaining 57 cosmetic `ruff` findings (line length, import order, `assertRaises(Exception)` in two tests) — not blocking, just housekeeping

## Change log (this pass)

- Fixed `pyproject.toml`: invalid `build-backend` (`setuptools.backends._legacy:_Backend`, doesn't exist) → `setuptools.build_meta`. Without this, `pip install -e .` failed outright.
- Fixed two real static-analysis findings in `cli.py` and `utils/image_utils.py`
  (undefined-name annotations for `PipelineConfig` and `torch.Tensor`, now
  properly deferred under `TYPE_CHECKING` / string annotations) — harmless at
  runtime today, but would break `mypy` and any IDE type-checking.
  Removed one unused variable (`results` in `cli.py`).
- Verified: AST-parses clean, all modules import clean, 38/38 unit tests pass.
