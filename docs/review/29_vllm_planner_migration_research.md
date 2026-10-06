# 29 — vLLM Planner migration: research, design, and what's still unverified

Prompted by a direct question: can the whole inference stack collapse
onto one shared environment using vLLM/TensorRT, for "sota" optimization?
Answer, with evidence: no, not as a whole — but a real, well-evidenced
partial upgrade exists for the Planner tier specifically. This doc
records the research trail and the implementation decisions it drove.

## 1. Why "one global env for everything" doesn't work

Three independent, verified blockers — not intuition:

1. **vLLM itself has an open, unresolved transformers-version conflict
   with the exact model already causing this project's Critic-tier
   split.** `vllm-project/vllm` issue **#39216**: vLLM 0.19.0 on PyPI
   pinned `transformers<5,>=4.56.0`, but Gemma 4 needs
   `transformers>=5.5.0` just to recognize `model_type=gemma4`. Adding
   vLLM to a "unified" env doesn't dissolve version conflicts, it
   relocates them. (This is fixed by 0.29.0 — see §3 — but the general
   lesson holds: a serving framework's own pins are a moving target, not
   a free pass.)
2. **The Sketch tier has no path onto vLLM or TensorRT.** It's a
   from-scratch MaskGIT-style transformer with a bespoke VQGAN tokenizer
   and Halton-order masked sampling — architecturally unlike both
   autoregressive decoding and diffusion denoising. Serving it through
   either framework would require a custom model plugin/kernel written
   from scratch, disproportionate to this project's scope.
3. **vLLM-Omni's diffusion path gives no benefit for this project's
   access pattern.** `Tongyi-MAI/Z-Image-Turbo` is confirmed on its
   "particularly verified" model list, but the diffusers backend adapter
   is documented as a "black-box adapter" with CFG-parallel execution,
   sequence-parallel execution, TeaCache/Cache-DiT acceleration, and
   continuous batching all explicitly unsupported — it just wraps plain
   diffusers as-is. Continuous batching is the one thing that would
   matter, and PRD §7.5 already correctly rules it out as a non-goal.
   `Qwen-Image-Edit-2511` (Polish-Quality) isn't even on the verified
   list — real risk of silent output divergence per vLLM-Omni's own
   stated caveat.

## 2. What IS a real win: vLLM for the Planner tier

- vLLM's own official PyPI release notes for **0.17.0** list "New
  architectures: Qwen3.5 (#34110)" — native support for the Gated
  DeltaNet hybrid architecture, not a wrapper: fused Triton kernels
  (ported from Flash Linear Attention), a hybrid KV-cache manager
  handling mixed linear/full-attention layers, CUDA graphs.
- The Planner is the one tier where this matters most: PRD §5.3 calls
  its loop "the loop that runs dozens of times per session" — the only
  tier where per-turn decode latency is felt on every interaction,
  unlike Polish (one shot per finalize) or Critic (rare, on-demand).
- This fits the existing architecture rather than fighting it: the
  `ModelBackend` interface already abstracts implementation details from
  the swap orchestrator (see Critic's subprocess isolation). A third
  isolated venv/subprocess for Planner is additive, not a redesign.

## 3. Verified dependency chain (real metadata, not search-result paraphrase)

- Cloned `huggingface/diffusers` directly (`@main`, shallow) and read its
  actual `setup.py`: declared floor is `transformers>=4.41.2`, no upper
  bound — confirms diffusers itself was never the source of the
  Planner/Critic conflict.
- Downloaded `vllm==0.29.0`'s real wheel and extracted its `METADATA`
  directly (not from docs): `Requires-Dist: transformers>=5.10.4` (no
  ceiling — the #39216 issue is resolved by this version) and
  `Requires-Dist: torch==2.13.0` (exact). `bitsandbytes` is **not**
  declared anywhere in this metadata, not even as an optional extra;
  `compressed-tensors==0.17.0` **is** a required core dependency —
  signaling AWQ/GPTQ/compressed-tensors as vLLM's well-supported,
  first-class quantization path, bnb as a secondary/unsupported one.
- `transformers==5.17.0` specifically (not just the bare `>=5.10.4`
  floor) is independently reported as measured working for Qwen3.5 on
  vLLM 0.29.0 on real hardware (community deployment report, RTX PRO
  6000) — the same "pin to a specifically verified version" discipline
  already used for `requirements-critic.txt`'s `transformers==5.5.0`.
- Ran a **real `pip install --dry-run`** resolving `vllm==0.29.0` +
  `torch==2.13.0` + `transformers==5.17.0` together: the entire
  transitive dependency tree (`compressed-tensors`, `triton`,
  `flashinfer-python`, `numba`, etc.) resolved with **zero conflicts**.
  This is the strongest verification available without real GPU
  hardware, and it's real, not asserted.
- Found `RedHatAI/Qwen3.5-9B-quantized.w4a16` as the recommended
  checkpoint: GPTQ W4A16 via `llm-compressor`, RedHatAI being
  `llm-compressor`'s maintaining org — comparable provenance quality to
  Unsloth's canonical Gemma-4 bnb checkpoint already trusted for Critic.
  A documented `vllm serve` command exists for a near-identical
  checkpoint from the same lineage, reporting >100% average accuracy
  recovery vs. the unquantized baseline. **Not independently
  re-verified by this project** — flagged as a recommendation to
  evaluate, per PRD Appendix C.4's own "verify at adoption, not
  announcement" discipline, not a silent swap.
- Directly read `ZImageImg2ImgPipeline.__call__`'s real source at a
  specific pinned diffusers commit and confirmed `strength: float = 0.6`
  is its actual default — cross-verifying a decision made in an earlier
  review pass against real source rather than trusting the docs example
  it was originally copied from.

## 4. Two real bugs caught and fixed before they ever ran

1. **Architecture mistake, caught mid-implementation**: the first draft
   of `planner_backend_vllm.py` called vLLM's Python API directly,
   in-process. This would have silently done nothing useful — the real
   `krisna_inference` service process runs under `/opt/venv-inference`'s
   interpreter, which does not (and should not) have `vllm` or
   `torch==2.13.0` installed. Fixed by rebuilding it as a subprocess
   client (`planner_worker_vllm.py` + `planner_backend_vllm.py`),
   mirroring `critic_backend.py`'s already-correct, tested
   spawn/JSON-lines/terminate-wait-kill pattern exactly.
2. **`gpu_memory_utilization` mismatch with this project's own VRAM
   ledger**: vLLM's default (`0.92`) greedily reserves ~92% of the
   *entire physical GPU* for its own KV-cache pool at load time,
   completely independent of and unaware of `VRAMLedger` (§7.2), which
   does its own admission bookkeeping against declared budgets. Left
   at the default, this backend would starve or OOM-crash the Sketch
   tier, which PRD §5.3 requires to be co-resident with the Planner in
   the idle/conversing state. Fixed by computing the fraction dynamically
   from `spec.vram_gb` (this project's own declared budget) divided by
   the REAL, live-probed total GPU memory — the one place a live probe
   is genuinely required, since vLLM's own API is fraction-of-real-
   memory, not ledger-relative — with a 15% safety margin matching
   `VRAMLedger`'s own `safety_margin_gb` concept.

## 5. Design decisions, stated explicitly

- **Subclass, don't duplicate.** `planner_backend.py` was refactored to
  expose a `_generate_raw()` extension point specifically so
  `PlannerBackendVLLM` inherits RAG retrieval, system-prompt
  construction, and the JSON generate-validate-retry loop unchanged.
  One implementation of that logic, used by both backends — they cannot
  silently drift apart from each other the way, e.g., `dataset.py`'s
  stale `caption_mix_ratio` default did in an earlier review pass.
- **No krisna_inference import in the worker script**, matching
  `critic_worker.py`'s own established reasoning exactly: it must stay
  standalone, runnable via subprocess under a separate interpreter with
  zero shared package dependency. RAG retrieval stays in the parent
  process instead, since `planner_rag.py` was verified stdlib-only (no
  torch/vllm) — safe to keep there rather than duplicating it.
- **Opt-in at every layer, not a silent default change**: `factory.py`
  only selects the vLLM backend when `KRISNA_PLANNER_BACKEND=vllm` is
  explicitly set; the Docker build only creates the third venv when
  `--build-arg INSTALL_VLLM_PLANNER=true` is passed. This matches the
  project's own discipline of never silently changing defaults or costs
  — extended here to the build system itself, since vLLM's wheel alone
  is ~316MB before its dependency tree.

## 6. What is genuinely NOT verified — stated plainly, not glossed over

This project has no GPU in its own environment. Everything above is
verified at the package-metadata and dependency-resolution level, plus
one independently-reported real-hardware measurement for Qwen3.5 +
vLLM 0.29.0 specifically. NOT verified by this project directly:

- That `RedHatAI/Qwen3.5-9B-quantized.w4a16` actually produces
  design-planning output quality comparable to the unquantized/NF4
  baseline for THIS project's specific prompts and RAG-augmented system
  prompt structure (not the general benchmarks it was evaluated against).
- That the subprocess worker's actual load time, first-token latency,
  and per-turn decode latency deliver a real, measurable improvement
  over the existing transformers+NF4 backend on the target hardware
  envelope (PRD §3: 16–24GB default, 12–16GB low-VRAM).
- That `gpu_memory_utilization`'s computed fraction, once vLLM is
  actually resident, leaves enough real headroom for the Sketch tier's
  own ~3.0GB to coexist without contention — the formula is correct
  arithmetic against declared budgets, but declared budgets for Sketch
  and Planner together summing to under the envelope has always been an
  arithmetic claim (§7.3), not a live-hardware-measured one, and adding
  a second, differently-behaved memory-management system (vLLM's own
  allocator) into that mix is new territory this project hasn't run.

If a future real-hardware pass finds any of the above doesn't hold, the
fix belongs in `planner_backend_vllm.py`/`requirements-planner.txt`
directly — this is an explicitly flagged, testable increment, not a
claimed-verified replacement.
