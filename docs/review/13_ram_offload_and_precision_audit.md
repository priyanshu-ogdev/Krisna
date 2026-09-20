# Phase 13 — RAM-Offload Upgrade for Polish-Default & Precision Drift Audit (all backends)

Follow-up to Phase 12 §5's open finding: *"`model_registry.py`'s
`POLISH_DEFAULT` `vram_gb=8.0` looks under-sized... needs a real
hardware measurement to confirm and correct."* This phase closes that
finding — not by measuring on real hardware (still not available in
this environment), but by tracing the actual root cause in code, which
turned out to be more specific and more fixable than "the number is
probably wrong": **the registry's `quantization` field for this tier
was factually false**, and the corrected number follows directly from
that, without needing a live measurement to be confident in it.

Also does the phase-by-phase pass the user asked for: checked every
other backend (Planner, Sketch, Polish-Quality, Critic) for the same
failure mode — declared registry quantization not matching what the
backend's `load()` actually does — since finding one instance is a
reason to check for others, not assume they're fine.

---

## §1 — Root cause: `POLISH_DEFAULT`'s declared quantization was wrong, not just its VRAM number

`model_registry.py`'s `REGISTRY[Tier.POLISH_DEFAULT]` declared
`quantization="NF4/NVFP4"`. Reading `polish_default_backend.py`'s
`load()`:

```python
pipe = ZImagePipeline.from_pretrained(
    self.model_id, torch_dtype=resolve_dtype(self.dtype), low_cpu_mem_usage=False
)
```

No `quantization_config`, anywhere. This backend has never applied NF4 —
it loads at `bfloat16` (the constructor's own default), full precision
for the format. The registry's `vram_gb=8.0` was sized for a 4-bit
model that doesn't exist in this codebase.

**Why the fix isn't "add NF4 quantization to match the registry"** —
checked `training/polish/train_dpo.py` before assuming that direction:

```python
torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
```

The LoRA this backend loads was trained against a bf16/fp16 base, never
NF4/QLoRA. A LoRA adapter's weights are only validated relative to the
full-precision activations they were trained against — retroactively
NF4-quantizing the inference-time base without ever having trained
against a quantized base would introduce a genuine, uncharacterized
train/inference precision mismatch. **bf16 at inference is the correct,
train-inference-synced choice; the registry's claim of NF4 was simply
wrong.**

This is the same failure mode `model_registry.py`'s own module docstring
already warned about generally ("see planner_backend.py's docstring for
a real example of this drifting apart once and getting caught") — just
a second, previously uncaught instance of it, on a different tier.

## §2 — Fix applied

**Documentation-accuracy fixes** (matching reality, not changing behavior):
- `model_registry.py`'s module docstring: corrected "NF4/DPO-fine-tuned
  for Default-Polish" → explains bf16 is deliberate and why.
- `REGISTRY[Tier.POLISH_DEFAULT].quantization`: `"NF4/NVFP4"` →
  `"bf16 (deliberately unquantized — matches LoRA training precision)"`.
- `REGISTRY[Tier.POLISH_DEFAULT].vram_gb`: `8.0` → `14.0`. This is
  arithmetic-derived, not measured: Z-Image-Turbo is 6B parameters
  (Tongyi-MAI, 2025) × 2 bytes/param (bf16) ≈ 12GB for the DiT alone,
  plus the text encoder, VAE, and LoRA adapter the model card doesn't
  break out separately. 14.0 is a conservative round-up, bounded above
  by Tongyi-MAI's own published "<16GB" full-pipeline guidance. Labeled
  as an estimate in the code comment, same convention this file already
  uses for `LOW_VRAM_REGISTRY`'s other entries — **still needs a real
  hardware measurement to become a confirmed number**, but is now
  arithmetic-grounded instead of silently wrong.
- `polish_default_backend.py`'s module + `load()` docstrings: explain
  why bf16 is correct here, cross-referencing this doc.

**Functional upgrade — the RAM-offload the user asked for:**
- `ZImageTurboBackend` gained an `enable_cpu_offload: bool = False`
  constructor param, mirroring `QwenImageEditBackend`'s exact pattern
  (`polish_quality_backend.py`) for consistency: `pipe.
  enable_model_cpu_offload()` when enabled, `pipe.to("cuda")` otherwise.
  `enable_model_cpu_offload()` was chosen over
  `enable_sequential_cpu_offload()` for the same reason Polish-Quality's
  module docstring already documents (the latter has a confirmed
  incompatibility with bnb NF4 — diffusers GH #10800) — that specific
  incompatibility doesn't actually apply here since this tier is never
  NF4-quantized, but using the same mechanism as the tier right next to
  it in the same file is worth more than a differently-justified offload
  call that only one tier uses.
- `model_registry.py`'s `LOW_VRAM_REGISTRY[Tier.POLISH_DEFAULT]` — new
  entry, `vram_gb=12.0, ram_gb=2.0`. Reasoning, stated in-code: Z-Image-
  Turbo's DiT transformer dominates its parameter count and can't itself
  be offloaded away mid-forward-pass, so `enable_model_cpu_offload()`'s
  real saving here is bounded by "everything except the DiT" (text
  encoder + VAE, modest — hence the ~2GB `ram_gb`), not the full
  14GB→8GB reduction the old (wrong) numbers implied. This is a smaller,
  more honest saving than Polish-Quality's offload gets, and the code
  comment says so explicitly rather than implying parity.
- `factory.py`: `real_backend_factory` now wires `_LOW_VRAM` through to
  `ZImageTurboBackend`'s `enable_cpu_offload` — it was previously
  excluded with the comment *"8.0GB already fits a 12GB target"*, which
  was the direct downstream consequence of §1's bug. Comment corrected.

**Known, disclosed limitation of the low-VRAM entry:** `12.0GB` fits a
`12.0GB` low-VRAM envelope with exactly zero headroom — the same
situation Phase 6/Phase 12 already found and accepted for the Critic
tier, mitigated the same way (`swap_orchestrator.py`'s opt-in
`vram_safety_margin_gb`). Not re-litigated here; same tradeoff, same
existing mitigation, now applying to a second tier.

## §3 — Verification

Updated every test that encoded the old, wrong assumption rather than
leaving them silently red or silently deleted:

- `tests/inference/test_low_vram_mode.py` — `POLISH_DEFAULT` moved from
  the "unchanged, small tier" test group into the "offload-reduced tier"
  group (renamed accordingly); the 12GB-envelope-fits-alone test already
  covered it generically and needed no change.
- `tests/inference/test_inference_factory.py` — added
  `test_low_vram_env_var_enables_offload_on_polish_default`, updated
  `test_default_mode_leaves_offload_disabled` to also check Polish-
  Default, renamed the now-inaccurately-named
  `test_planner_and_polish_default_never_offloaded...` (Polish-Default
  IS offloaded now) to `test_planner_never_offloaded...`.
- `tests/inference/test_vram_budget.py` — three tests had the old 8.0GB
  figure baked directly into their arithmetic (envelope sizes chosen so
  a specific fit/no-fit outcome held at 8.0GB). Recomputed each by hand
  against 14.0GB and adjusted envelope constants where the *outcome*
  the test was actually checking (not just the number) would otherwise
  flip for the wrong reason (`test_release_frees_budget`'s envelope
  15.0→22.0, keeping "doesn't fit before release, fits after" intact).
- `tests/inference/test_swap_orchestrator.py` —
  `test_vram_budget_preflight_skips_straight_to_fallback`'s envelope
  (10.0→15.0) for the same reason: the test's actual point (Polish-
  Quality doesn't fit, falls back to Polish-Default, which does) needs
  an envelope between the two tiers' real sizes, and 14.0 sits above the
  old fallback envelope.

**Full `tests/inference` suite: 124 passed** (up from 123 before this
phase — one new test added, none skipped, none newly failing).

## §4 — Phase-by-phase precision-drift check, all other backends

Since one real instance of "declared registry quantization ≠ actual
backend behavior" was found, checked every other tier's `load()` against
its registry claim rather than assuming the rest are fine.

| Tier | Registry claim | Backend's actual `load()` | Match? |
|---|---|---|---|
| Planner | `4-bit (fast mode: Qwen3.5-4B available)` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`, applied whenever `quantize=True` (the default) | **Yes** — verified via `load_kwargs["quantization_config"]` is genuinely set, not just documented. This backend's own docstring already documents a prior, self-caught version of exactly this bug class (an earlier revision defaulted to bf16) — confirms the fix from that prior catch is still in place. |
| Sketch | `n/a (small, from-scratch)` | `MaskGITSketchModel.from_checkpoint(...)`, no quantization anywhere | **Yes** — no quantization claimed, none applied. |
| Polish-Default | `NF4/NVFP4` | `torch_dtype=resolve_dtype(self.dtype)` (bf16), no `quantization_config` | **No — this phase's fix.** |
| Polish-Quality | `NF4 (fully frozen — no adapter)` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` passed as `quantization_config` | **Yes** — verified genuinely applied. |
| Critic | `NF4 (fully frozen...)` | Pre-quantized checkpoint (`unsloth/gemma-4-31B-it-unsloth-bnb-4bit`) + `load_in_4bit=True` in both the unsloth and plain-transformers offload code paths | **Yes** — verified both branches, not just the primary one. |

**Result: one mismatch found (Polish-Default, now fixed), four tiers
confirmed genuinely consistent** between their registry claim and their
actual loading code. This was a targeted, code-read verification (not a
live GPU run) — the same limitation every prior phase in this review has
disclosed applies here too: correct on paper, not yet exercised against
real weights in this environment.

## §5 — Inference pipeline & flow integrity, re-checked after the offload wiring

The user's ask specifically included "maintaining proper inference
pipeline and flow" — checked that adding the offload path didn't change
anything about the tier's actual contract with the orchestrator:

- `run()`'s signature, its `handoff_image_ref`/`constraints`/`prompt`/
  `seed` handling, `guidance_scale=0.0` (required for Turbo — unrelated
  to and unaffected by the offload change), and the returned
  `{tier, image_ref, verifier_scores}` shape are **all unchanged** —
  `enable_cpu_offload` only affects `load()`'s device-placement call,
  nothing about how a loaded pipeline is invoked or what it returns.
- `unload()` is unchanged — `enable_model_cpu_offload()`-managed
  pipelines still respond correctly to `self._pipe = None` +
  `torch.cuda.empty_cache()` (diffusers' offload hooks are attached to
  pipeline submodules, not external state this backend tracks
  separately).
- `SwapOrchestrator`'s admission/swap sequence (`swap_orchestrator.py`)
  is untouched by this phase — it only ever reads `ModelSpec.vram_gb`/
  `ram_gb` from whichever registry `get_registry(low_vram=...)` returns,
  which is exactly the mechanism this phase's `LOW_VRAM_REGISTRY` entry
  plugs into. No orchestration code needed to change for this fix,
  which is itself a confirmation that the tier-boundary contract
  (registry declares budget, orchestrator enforces it, backend receives
  a flag derived from which registry is active) is sound — a bug in one
  tier's declared numbers didn't require touching the mechanism that
  reads them.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Resolved (closes Phase 12's open item)** | `POLISH_DEFAULT`'s declared `quantization="NF4/NVFP4"` was factually false — backend has never applied it, by design (train/inference precision match) — and the `vram_gb=8.0` figure inherited that false assumption | Fixed: quantization label corrected, `vram_gb` recomputed from real bf16 arithmetic (14.0, still an estimate pending real-hardware measurement), RAM-offload path added and wired through `factory.py`, `LOW_VRAM_REGISTRY` entry added |
| Info | Checked all four other backends for the same drift class | None found — Planner, Sketch, Polish-Quality, Critic all confirmed consistent between declared and actual quantization |
| Info | Inference pipeline contract (`run()`/`unload()` signatures, orchestrator's registry-driven admission) confirmed unaffected by the offload addition | No action needed — the fix is fully contained to `load()` and the registry numbers it's checked against |

## Still open (unchanged from Phase 12)
- Real hardware measurement of Polish-Default's actual resident VRAM
  (both offloaded and not) — this phase makes the estimate
  arithmetic-grounded instead of wrong, but "grounded estimate" and
  "measured" remain different confidence levels.
- Phase 1's RICO join/count verification (top open item since Phase 1,
  unaffected by this phase).
