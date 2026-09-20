# Phase 6 — Inference Orchestrator (SwapOrchestrator, VRAM/RAM budget, model registry)

> **RESOLVED (post-Phase-6 follow-up session):** the "Low" severity
> zero-headroom finding below is now correctly documented in code —
> `vram_budget.py`'s `_BaseLedger.safety_margin_gb` docstring previously
> claimed `VRAMLedger` sets a non-zero default, which was simply false
> (no such override existed anywhere, confirmed by grep across the whole
> `inference/` tree). The comment has been corrected to state the actual,
> deliberate design: both ledgers default to `0.0`, opt-in only, because
> a nonzero default would make the Critic low-VRAM tier permanently
> inadmissible. See `docs/review/12_post_upgrade_resync_audit.md` §2.
> This was a documentation-accuracy bug, not a behavior bug — the
> zero-headroom sizing itself was already correct as designed.

## Architecture — sound design, verified against real files
- `_BaseLedger` (shared accounting for `VRAMLedger`/`RAMLedger`) is a genuine shared base class, not copy-pasted twice — a fix to admit/release/would_fit logic fixes both ledgers at once. Real, verified in code, not just claimed in a docstring.
- Admission decisions are based on the **declared** ledger (`ModelSpec.vram_gb`), not a live `torch.cuda` probe — `probe_real_vram()`/`probe_real_ram()` exist purely for diagnostics/logging, explicitly never used for admission. This is the right call for CI-testability and deterministic behavior; verified this separation is real, not just asserted.
- OOM-handling policy (`swap_orchestrator.py`'s own docstring, verified against actual code): check `would_fit()` before attempting a load → real OOM retry with backoff → `fallback_tier` (POLISH_QUALITY → POLISH_DEFAULT only; CRITIC has none) → baseline (Planner+Sketch) residency restoration with *more* retries than a normal load, since that's the one unrecoverable failure mode. This is a coherent, defensible design, and the asymmetric extra-retries-for-baseline-restoration detail is a genuinely good touch worth citing as evidence of real engineering care, not boilerplate.
- `get_registry(low_vram=...)` as the single source of truth for which VRAM/RAM numbers are active (rather than scattered `if low_vram` checks) is a real anti-drift design choice, verified as the actual single call site consumers use.

## Correction — my original "real bug" finding was wrong, verified and retracted

**This section corrects Phase 6's original headline finding, found while
attempting to actually fix it.** I originally computed always-resident
baseline (Planner 6.5 + Sketch 3.0 = 9.5GB) as a permanent floor, added
Polish Default's unchanged 8.0GB low-VRAM footprint, got 17.5GB against
a 12GB low-VRAM envelope, and called it a critical bug.

That was wrong. `swap_orchestrator.py`'s `_finalize_locked` and
`_critique_locked` both call `await self._unload_baseline()` **before**
loading the swappable tier, with an explicit comment confirming this is
intentional: *"Unload baseline before loading the polish tier — this is
the step that makes the 16-24GB envelope achievable at all."* The
system never holds baseline + a swappable tier resident at the same
time — it alternates between "conversation mode" (baseline only, 9.5GB)
and "finalize/critique mode" (one swappable tier only, baseline
unloaded first). Under that actual behavior, every low-VRAM tier fits
its own envelope independently: Polish Default 8.0GB ≤ 12GB ✓, Quality
10.0GB ≤ 12GB ✓, Critic 12.0GB ≤ 12GB ✓ (exact fit, zero headroom —
worth a small safety margin in practice, but not a functional bug).

**Retracting the "Real bug" severity finding below** and replacing it
with the corrected, much lower-severity note about zero headroom on the
Critic tier. I'm leaving the original incorrect writeup struck through
rather than deleted, since an honest review should show its own
mistakes, not just quietly fix them.

~~**The declared low-VRAM registry cannot actually fit inside the
declared low-VRAM envelope.**~~ — **incorrect, see correction above.**
The original reasoning (baseline treated as a permanent, un-evicted
floor) doesn't match the actual `_unload_baseline()`/`_restore_baseline()`
behavior around every finalize/critique call.

**What's actually still worth flagging, at low severity:** the Critic's
low-VRAM footprint (12.0GB) fits the 12.0GB envelope with **exactly
zero headroom**. Any real-world overhead not captured in the declared
`vram_gb` estimate (CUDA context, fragmentation, activation memory
during generation) would push this over budget in practice, even though
the declared numbers technically balance. Recommend a small explicit
safety margin (e.g. envelope check against `envelope_gb - safety_margin_gb`)
rather than exact equality being treated as "fits."



## Secondary items, verified consistent (not re-litigated at length)
- Planner's `vram_gb=6.5` (NF4, 9B) is internally consistent with Phase 3's independently-verified NF4-default finding — cross-phase consistency confirmed, not just assumed twice.
- Critic's low-VRAM RAM estimate (`ram_gb=40.0`) is a **computed** lower bound (10B params × 4 bytes FP32 CPU-offload residual, per bitsandbytes' documented `llm_int8_enable_fp32_cpu_offload` behavior), not a guess — the comment shows the actual arithmetic, and it's the right order of magnitude for that offload mechanism. Still flagged in-repo as needing live-run validation, appropriately.
- `POLISH_QUALITY`'s adapter-framing bug (`"NF4 (frozen backbone)"` → `"NF4 (fully frozen — no adapter)"`) is confirmed fixed in the current registry text — a real fix, verified present, not just claimed.

## Findings summary

| Severity | Finding |
|---|---|
| ~~**Real bug**~~ **Retracted** | ~~Low-VRAM envelope math broken~~ — **incorrect, see correction section above.** Baseline is temporarily unloaded before every finalize/critique, so no sum-fit issue exists. |
| Low | Critic's low-VRAM footprint (12.0GB) fits its 12.0GB envelope with exactly zero headroom — recommend a small explicit safety margin rather than relying on exact equality |
| Info | Ledger/OOM-handling architecture is sound and verified in code, not just asserted in docstrings |
| Info | Cross-phase consistency confirmed for Planner's VRAM figure (matches Phase 3's independent finding) |
| Info | Critic's RAM-offload estimate is a real computed lower bound, correctly still flagged as needing live validation |

## Citations (Phase 6)
No new external citations — this phase is systems/engineering design review, not model-technique review. bitsandbytes' documented `llm_int8_enable_fp32_cpu_offload` behavior (referenced for the Critic RAM-offload estimate) should be cited to the bitsandbytes documentation directly if this arithmetic goes in the paper's appendix.

---
Next: Phase 7 — Consolidated citations doc, merging every citation from Phases 1–6 with the repo's existing `RESEARCH_AND_CITATIONS.md` into one paper-ready bibliography, plus a rollup of every open action item across all phases.
