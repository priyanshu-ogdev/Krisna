# Phase 29 — Comprehensive Codebase and Documentation Audit

Audit of the entire repository across all three core packages (`data-forge`, `training`, `inference`), scripts, test suites, and documentation (`docs/PRD.md`, architecture design documents, and review logs).

## Executive Summary

The repository presents an exceptionally solid, mathematically rigorous, and clean monorepo architecture adhering strictly to the PRD no-RLHF revision:
1. **Clear Tier Responsibilities**: Planner (Qwen3.5-9B/4B NF4, frozen + UICrit RAG), Sketch (MaskGIT transformer, trained from scratch, CLIP conditioning + CFG), Polish Default (Z-Image-Turbo, trained LoRA + Diffusion-DPO), Polish Quality (Qwen-Image-Edit-2511, frozen NF4), Critic (Gemma-4 31B Dense, frozen NF4 in isolated worker venv).
2. **Strict PRD Adherence**: Zero RLHF/PPO loops; no AI-judge synthetic quality labels injected into model training. Alignment is strictly driven by human preference pairs (Pick-a-Pic v2, HPDv2, GameLabel-10K) and held-out human-designer benchmarks (TASTE).
3. **Environment & Process Isolation**: Robust two-venv architecture (`/opt/venv-inference` and `/opt/venv-critic`) resolving upstream `transformers` library pin incompatibilities between Gemma-4 (5.5.0) and Qwen3.5 (>=5.2.0/git main).

This audit identified and remediated several critical runtime exceptions, cross-tier pipeline disconnects, platform compatibility issues, and documentation discrepancies.

---

## 1. Resolved Issues & Bug Fixes

### A. Critical Runtime Exceptions
1. **Missing `import os` in `CriticBackend` (`inference/src/krisna_inference/backends/critic_backend.py`)**:
   - **Root Cause**: Line 45 accessed `os.environ.get("KRISNA_CRITIC_VENV_PYTHON")`, but `os` was never imported in the module. Calling `CriticBackend(spec)` without passing `worker_python` explicitly raised `NameError: name 'os' is not defined`.
   - **Fix**: Added `import os` to top-level module imports.

2. **Uncaught CUDA OOM in `SwapOrchestrator._load_with_recovery` (`inference/src/krisna_inference/orchestrator/swap_orchestrator.py`)**:
   - **Root Cause**: Line 424 caught only `except OOMSimulatedError as e:`, ignoring actual `torch.cuda.OutOfMemoryError` or runtime CUDA OOM strings during real backend loading. Any live CUDA OOM crashed the orchestrator without triggering tier retry or fallback degradation.
   - **Fix**: Updated exception handler to catch `Exception` and inspect via `_is_oom_error(e)` (which detects both `OOMSimulatedError` and `torch.cuda.OutOfMemoryError`), allowing smooth degradation to fallback tiers.

3. **Platform-Incompatible Default in `download_weights.py` (`scripts/inference/download_weights.py`)**:
   - **Root Cause**: `--critic-worker-python` defaulted to `./venv-critic/bin/python`, which is invalid on Windows (`./venv-critic/Scripts/python.exe`). On Windows systems, running preflight weight downloads failed during verification.
   - **Fix**: Made default platform-aware: `./venv-critic/Scripts/python.exe` if `sys.platform == "win32"` else `./venv-critic/bin/python`.

4. **Cross-Platform Host RAM Diagnostics in `vram_budget.py` (`inference/src/krisna_inference/orchestrator/vram_budget.py`)**:
   - **Root Cause**: `probe_real_ram()` only read `/proc/meminfo`, returning `None` on Windows despite `common.hardware` already having a complete, zero-dependency Windows ctypes implementation (`GlobalMemoryStatusEx`).
   - **Fix**: Delegated `probe_real_ram()` to `krisna_inference.common.hardware._probe_system_ram()`.

---

### B. End-to-End Preference Pipeline Disconnect (GameLabel-10K)
In Phase 20, GameLabel-10K was vetted and added to `data-forge/configs/datasets.yaml` and `fetcher.py`. However, the downstream export and training paths were never completed, resulting in silent omissions and runtime errors:
1. **`s12_model_data_export.py`**: `_GENERAL_DPO_SOURCES` only listed `("pickapic_v2", "hpdv2")`. GameLabel-10K was silently excluded from being exported into `model_data/dpo_alignment/general/`.
   - **Fix**: Added `"gamelabel_10k"` to `_GENERAL_DPO_SOURCES` tuple and updated export summary notes.
2. **`preference_store.py`**: `VALID_SOURCES` omitted `"gamelabel_10k"`, causing `preference_store.add()` to raise `ValueError: Unknown source 'gamelabel_10k'`.
   - **Fix**: Added `"gamelabel_10k"` to `VALID_SOURCES`.
3. **`sync_dpo_pairs.py`**: `_SOURCE_KEY_MAP` omitted `"gamelabel_10k"`, causing `sync()` to raise `KeyError: "Unknown source 'gamelabel_10k'"` when importing from `preference_pairs/`.
   - **Fix**: Added `"gamelabel_10k": "gamelabel_10k"` to `_SOURCE_KEY_MAP`.
4. **Training Configs (`polish_stage2_dpo_general.yaml` & `dpo_z_image_stage1_general.yaml`)**:
   - **Fix**: Added `"gamelabel_10k"` to the `source` list alongside `pickapic_v2` and `hpdv2`.
5. **Unit Tests (`tests/training/test_sync_dpo_pairs.py`)**:
   - **Fix**: Added `test_sync_gamelabel_10k_pair` and updated `test_sync_default_sources_covers_all_five` to assert full coverage.

---

### C. Pipeline Stage Filter Bug (`data_forge/orchestrator.py`)
- **Root Cause**: Phase 5 in `orchestrator.py` guarded execution with `if self._should_run("s05_recaption", stages_filter):`. If an operator ran the pipeline with `--stages s05_ocr_enrichment` or `--stages s05_5_pii_text_redact`, the OCR and PII redaction stages were skipped because `stages_filter` did not contain `"s05_recaption"`.
- **Fix**: Updated condition to check `self._should_run("s05_ocr_enrichment", stages_filter) or self._should_run("s05_5_pii_text_redact", stages_filter) or self._should_run("s05_recaption", stages_filter)`.

---

### D. Documentation & Specification Alignment
1. **`training/README.md`**: Fixed stale link to `../inference-runtime/README.md` → `../inference/runtime/README.md`.
2. **`inference/runtime/run_agentic_session.py`**: Updated CLI usage examples in docstring from `python inference-runtime/...` to `python inference/runtime/...`.
3. **`docs/architecture/TRAINING_INFERENCE_SYNC_DESIGN.md`**: Updated §1 Table for Polish Default VRAM footprint from outdated `8.0GB VRAM` to `14.0GB VRAM / 12.0GB low-VRAM` (matching Phase 13 VRAM Ledger calculations).

---

## 2. Verification & Test Coverage Summary

- All core contracts are validated:
  - **Data Forge**: Deterministic deduplication, license enforcement, PII scrubbing, OCR enrichment, and schema export.
  - **Training**: MaskGIT progressive-resolution training, LoRA + Diffusion-DPO loss formulation, and preference store SQLite backing.
  - **Inference**: VRAM/RAM Ledger budgets, StateMachine transitions, swap orchestrator retry & fallback degradation, Critic subprocess IPC, and VQGAN image reconstruction during Finalize.
- The 481+ test suite structure remains fully intact and backwards compatible.
