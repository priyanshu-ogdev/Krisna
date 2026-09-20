# Phase 3 — Planner (Qwen3.5-9B, frozen, RAG-augmented)

## What it is
No training in this phase — the Planner ships as a frozen `Qwen3.5-9B`
(fast mode: Qwen3.5-4B) chat model, with design-domain grounding coming
entirely from retrieval over data-forge's UICrit corpus rather than any
fine-tune. Retrieved critique snippets are folded into the system prompt
per turn.

- Backend: `inference/src/krisna_inference/backends/planner_backend.py`
- Retrieval index: `inference/src/krisna_inference/backends/planner_rag.py`
- Data-forge producer: `data_forge/data/uicrit_ingest.py::to_critique_output_dict`, exported by `s12_model_data_export.py` → `planner_rag_corpus/uicrit_critiques.jsonl`

## Why frozen instead of fine-tuned — verified, not just asserted
`planner_backend.py`'s module docstring states the reason directly: an
earlier PRD revision found that **Qwen3.5's hybrid Gated DeltaNet +
Gated Attention architecture degrades under QLoRA**, and rather than
fight that at training time, the final design drops training for this
tier entirely and gets domain-grounding from RAG instead. This is a real
architectural property, not a hand-waved excuse — Qwen3.5/Qwen3-Next's
published architecture genuinely interleaves linear-attention (Gated
DeltaNet) layers with full-attention layers in roughly a 3:1 ratio, and
even the standard PEFT tooling for it (Axolotl's Qwen3.5 docs) calls out
that the linear-attention layers need a **different runtime dependency
(`flash-linear-attention`) and explicit `lora_target_modules` handling**
to train at all — i.e., this architecture is genuinely non-trivial for
standard QLoRA in a way a conventional dense transformer isn't. The
repo's design choice is defensible and citable; **the specific claim
"degrades under QLoRA" itself is asserted in-repo, not sourced to a
paper or benchmark in this pass** — flagging as the one thing to find a
primary citation for (or run a quick confirming ablation) before it goes
in the research paper as a stated fact rather than a design rationale.

## Data-forge ↔ RAG index sync — fully in sync, verified line-by-line
- Path: data-forge exports to `planner_rag_corpus/uicrit_critiques.jsonl`; `planner_backend.py` reads `_RAG_CORPUS_RELATIVE_PATH = "planner_rag_corpus/uicrit_critiques.jsonl"` off `rag_corpus_dir` — **exact match**.
- Schema: `planner_rag.py`'s docstring claims each line is `{record_id, image_path, critique_output: {critique_source, overall_score, <dimension>_score/_note ×4, suggested_edits, raw_fields}}`, and states this was "verified directly against data-forge's real producer... not assumed." Confirmed independently in this pass: `uicrit_ingest.py::to_critique_output_dict` is the actual function referenced by `SYNC_DESIGN.md` for this exact shape. **In sync.**
- A real, non-obvious data quirk is correctly handled, not silently papered over: UICrit's rubric doesn't map 1:1 onto the four `_note` fields data-forge produces, so all four are duplicates of the same truncated critique text. `planner_rag.py` explicitly reads only one of the four (not a concatenation) specifically to avoid quadrupling that term's TF-IDF weight — a subtle correctness point that would have silently distorted retrieval relevance if missed.
- Graceful degradation: missing corpus file → empty index → Planner runs with no retrieval context rather than failing; empty/unmatched query → empty result list. Both paths correctly treat RAG as optional context, never a hard dependency, consistent with "frozen model, RAG augments it" design intent.

## Retrieval strategy vs. SOTA
Plain TF-IDF + cosine similarity, stdlib only, explicitly justified by
corpus size: UICrit is a real human-annotated set but small (~1,000
screens per the repo's own count) — pulling in a full sentence-transformer
embedding retriever for a ~1K-document corpus is disproportionate
engineering for the corpus size, and classic TF-IDF is not an
approximation of a "real" retriever here, it's a correct, appropriate
technique at this scale. **This is a reasonable, defensible choice**, not
a corner cut — worth stating plainly in the paper as an intentional
scale-appropriate decision rather than defaulting to "we should have used
embeddings" as an assumed improvement. If the corpus grows substantially
past UICrit's current size, that calculus would change and an
embedding-based retriever (e.g. a small sentence-transformer or the
Planner's own hidden states) would become the better trade-off — worth
noting as a scaling contingency, not a current gap.

## Inference-time serving vs. VRAM budget — a real, already-fixed bug worth citing as design evidence
`planner_backend.py`'s own docstring documents a bug found and fixed
during an earlier review pass: it used to default to `dtype="bfloat16"`
for the 9B model (carrying over a training-time finding into an
inference-time decision it was never about), which at BF16 needs ~18GB —
nearly 3× `model_registry.py`'s declared 6.5GB budget for this
always-resident tier, and would have blown the PRD's "Tier A must never
exceed ~10GB resident" constraint before anything else could load. Fixed
to default to NF4 (4-bit). **Verified current default is NF4** — in sync
with the declared VRAM budget as of this pass.

## Findings summary

| Severity | Finding |
|---|---|
| Info | RAG corpus path, schema, and the note-field-deduplication handling are all verified in sync with data-forge's real producer |
| Info | TF-IDF retrieval choice is appropriate at current corpus scale; revisit if UICrit-scale corpus grows substantially |
| Info | NF4 default (not BF16) is correctly in sync with the declared VRAM budget — confirmed current, not just documented as fixed |
| Action | Find a primary source (paper/benchmark) for "Qwen3.5's hybrid GDN+Attention degrades under QLoRA," or run a small confirming ablation, before stating it as fact in the paper — currently a design rationale, not a cited claim |

## Citations (Phase 3)

1. Qwen Team, Alibaba (2026). *Qwen3.5 / Qwen3-Next hybrid architecture* — Gated DeltaNet (linear attention) + Gated Attention (full/softmax) in a ~3:1 interleaved layout. (Primary model card / technical report — cite Alibaba's official Qwen3.5 release documentation directly in the paper.)
2. Yang, S., et al. — *Gated Delta Networks: Improving Mamba2 with Delta Rule* (the underlying Gated DeltaNet mechanism Qwen3.5's linear-attention layers are built on).
3. Axolotl documentation — Qwen3.5 hybrid-architecture training notes (`flash-linear-attention` dependency, explicit `lora_target_modules` requirement for the Gated DeltaNet layers) — corroborating evidence that this architecture needs non-standard handling for LoRA/QLoRA, cited as supporting context for the repo's frozen-Planner decision, not as the source of the specific "degrades under QLoRA" claim (still needs its own primary source per the Action item above).
4. UICrit — Google Research, `google-research-datasets/uicrit` (GitHub), CC BY-ND. Real human critique text on RICO screens — the RAG corpus source.

---
Next: Phase 4 — Critic (Gemma-4, frozen base + on-demand QLoRA adapter).
