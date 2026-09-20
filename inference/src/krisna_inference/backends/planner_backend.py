"""Planner/Critic-conversational backend — Qwen3.5-9B (fast mode: Qwen3.5-4B).

FROZEN MODEL (final, no-RLHF-loop PRD revision): this project never
fine-tunes the Planner. §6.1's finding — that Qwen3.5's hybrid Gated
DeltaNet + Gated Attention mechanism degrades under QLoRA — is about
TRAINING-time behavior specifically, and is now moot for this tier: there
is no training happening on this model at all. Design-domain grounding
comes from retrieval over data-forge's real, human-annotated UICrit
corpus (see planner_rag.py) instead — retrieved critique snippets are
folded into the system prompt per turn.

BUG FIX (found reviewing against model_registry.py's REGISTRY entries):
an earlier revision of this backend defaulted to `dtype="bfloat16"`,
carrying the training-time §6.1 finding over into an inference-time
decision it was never actually about. BF16 for a 9B model needs ~18GB —
nearly 3x model_registry.py's declared `vram_gb=6.5` budget (sized for
4-bit) for this `always_resident=True` tier, and would alone blow past
PRD §5.3's load-bearing "Tier A must never exceed ~10GB resident" rule
before the sketch tier or anything else could load in the idle/
conversing state. Now defaults to NF4 (4-bit), matching the declared
budget and PRD §7.3's resident-footprint table ("Planner (4-bit) +
sketch tier + verifiers ~8-10GB").

This is NOT the same as claiming 4-bit inference quality is validated —
PRD §15 leaves that genuinely open ("FP8 serving safety for Qwen3.5-9B...
inference-time quantization not yet independently validated"). If a real
evaluation later finds 4-bit inference unacceptably degrades planning
quality, the fix is to raise model_registry.py's declared vram_gb budget
to match BF16 (~18GB) and re-check the whole idle-state VRAM sum against
the 24GB envelope — not to silently reintroduce BF16 here without also
updating the budget it would violate.

Structured output (the JSON design-state delta the orchestrator's flows.py
expects) uses a generate-validate-retry loop rather than a hard grammar
constraint: it's simpler, needs no additional generation-time dependency
(no verified support for constrained decoding against this specific
tokenizer/attention combination was found), and is transparent about
failure — a malformed response after MAX_JSON_RETRIES attempts surfaces a
distinct error rather than either silently returning free text mislabeled
as structured JSON or crashing.

Requires `transformers` built from git main (Qwen3.5 support isn't in a
tagged PyPI release as of this build — see huggingface.co/Qwen/Qwen3.5-9B),
plus `torchvision`/`pillow` (it's a VL model) and, for full-speed Gated
DeltaNet, the optional `causal_conv1d` + `flash-linear-attention` kernels.
Without those two kernel packages the linear-attention layers silently fall
back to slow PyTorch ops rather than failing — this backend checks for them
at load time and logs a warning rather than letting that surprise show up
as unexplained latency later.
"""

from __future__ import annotations

import json
import logging

from krisna_inference.backends.common import pick_device, resolve_dtype
from krisna_inference.backends.planner_rag import UICritRAGIndex
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.planner")

MAX_JSON_RETRIES = 2
_RAG_CORPUS_RELATIVE_PATH = "planner_rag_corpus/uicrit_critiques.jsonl"


class PlannerJSONDecodeError(Exception):
    """Raised when the model fails to produce valid JSON after retries.
    Distinct from a generic RuntimeError so callers (flows.py) can decide
    how to degrade — e.g. fall back to a free-text-only turn — rather than
    treating it the same as an OOM or a load failure."""


class PlannerBackend(ModelBackend):
    def __init__(
        self,
        spec,
        model_id: str = "Qwen/Qwen3.5-9B",
        fast_model_id: str = "Qwen/Qwen3.5-4B",
        use_fast_mode: bool = False,
        rag_corpus_dir: str | None = None,
        dtype: str = "bfloat16",  # compute dtype for NF4's bnb_4bit_compute_dtype, not the storage dtype
        quantize: bool = True,     # NF4 by default — see module docstring's budget-match note
        rag_top_k: int = 3,
        max_gpu_gb: float | None = None,   # CPU-offload cap — see module docstring's
        max_cpu_gb: float | None = None,   # low-VRAM-mode section. Both None = no offload
                                             # (current behavior, single-GPU device_map="auto").
                                             # Not needed for the Planner in practice — its 6.5GB
                                             # NF4 footprint already fits a 12GB target on its own
                                             # — but wired through consistently with the other
                                             # backends in case a smaller-than-12GB target is ever
                                             # attempted, or a future larger Planner checkpoint needs it.
    ) -> None:
        super().__init__(spec)
        self.model_id = fast_model_id if use_fast_mode else model_id
        self.rag_corpus_dir = rag_corpus_dir
        self.dtype = dtype
        self.quantize = quantize
        self.rag_top_k = rag_top_k
        self.max_gpu_gb = max_gpu_gb
        self.max_cpu_gb = max_cpu_gb
        self._model = None
        self._tokenizer = None
        self._rag_index: UICritRAGIndex | None = None

    async def load(self) -> None:
        import asyncio

        def _load_sync():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._check_kernels()

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            load_kwargs = {"device_map": "auto"}
            if self.quantize:
                from transformers import BitsAndBytesConfig

                bnb_kwargs = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": resolve_dtype(self.dtype),
                }
                if self.max_gpu_gb is not None:
                    # CONFIRMED real, documented requirement (not guessed):
                    # bitsandbytes raises "Some modules are dispatched on the
                    # CPU or the disk..." unless this flag is set alongside
                    # max_memory — see module docstring's low-VRAM-mode
                    # section for the real-world RAM-cost implication (CPU-
                    # offloaded portion is stored in FP32, not 4-bit — an 8x
                    # per-parameter size increase relative to NF4).
                    bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
                    load_kwargs["max_memory"] = {
                        0: f"{self.max_gpu_gb}GiB",
                        "cpu": f"{self.max_cpu_gb or 64}GiB",
                    }
                load_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
            else:
                load_kwargs["dtype"] = resolve_dtype(self.dtype)

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            # REMOVED: LoRA adapter loading (PeftModel.from_pretrained).
            # The Planner ships frozen — see module docstring. If a future
            # PRD revision un-freezes it, restore adapter loading here
            # explicitly rather than silently reintroducing it.
            model.eval()
            return model, tokenizer

        try:
            self._model, self._tokenizer = await asyncio.to_thread(_load_sync)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                raise OOMSimulatedError(str(e)) from e
            raise BackendLoadError(f"Failed to load planner '{self.model_id}': {e}") from e

        if self.rag_corpus_dir:
            from pathlib import Path

            corpus_path = Path(self.rag_corpus_dir) / _RAG_CORPUS_RELATIVE_PATH
            self._rag_index = await asyncio.to_thread(UICritRAGIndex.from_jsonl, corpus_path)
        else:
            log.warning(
                "planner_no_rag_corpus_configured",
                extra={"note": "No rag_corpus_dir set — planner will run with no retrieval "
                                "context at all. Set KRISNA_PLANNER_RAG_CORPUS_DIR to "
                                "data-forge's model_data/ directory."},
            )
            self._rag_index = UICritRAGIndex()  # empty index, degrades cleanly

        self._loaded = True

    def _check_kernels(self) -> None:
        missing = []
        try:
            import causal_conv1d  # noqa: F401
        except ImportError:
            missing.append("causal_conv1d")
        try:
            import fla  # noqa: F401
        except ImportError:
            missing.append("flash-linear-attention (fla)")
        if missing:
            log.warning(
                "planner_missing_fast_kernels",
                extra={
                    "missing": missing,
                    "note": "Gated DeltaNet layers will fall back to slow PyTorch "
                    "ops without these — install for real per-turn latency.",
                },
            )

    async def unload(self) -> None:
        import asyncio

        def _unload_sync():
            import gc

            self._model = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        await asyncio.to_thread(_unload_sync)
        self._rag_index = None
        self._loaded = False

    async def run(
        self,
        message: str = "",
        constraints: dict | None = None,
        conversation_history: list[dict] | None = None,
        prior_critique: dict | None = None,
        **kwargs,
    ):
        if not self._loaded:
            raise RuntimeError("Planner backend not loaded")
        import asyncio

        retrieved = self._rag_index.retrieve(message, k=self.rag_top_k) if self._rag_index else []
        system = self._build_system_prompt(constraints or {}, retrieved, prior_critique=prior_critique)

        def _generate(extra_instruction: str | None = None) -> str:
            import torch

            messages = [{"role": "system", "content": system}]
            if conversation_history:
                for turn in conversation_history[-6:]:
                    role = "assistant" if turn.get("role") == "planner" else turn.get("role", "user")
                    content = turn.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})
            if extra_instruction:
                messages.append({"role": "system", "content": extra_instruction})
            messages.append({"role": "user", "content": message})

            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            return self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

        def _run_sync() -> dict:
            # Generate-validate-retry loop, not a hard grammar constraint —
            # see module docstring for why. First attempt asks for free-text
            # + a trailing JSON delta; on a parse failure, retry with a
            # sharper instruction rather than silently accepting garbage or
            # crashing on the first miss.
            text = _generate()
            delta, error, span = self._extract_json_delta(text)
            attempts = 1
            while delta is None and attempts <= MAX_JSON_RETRIES:
                text = _generate(
                    extra_instruction=(
                        "Your previous reply did not end with a valid JSON object "
                        f"matching the required design-state-delta shape ({error}). "
                        "Reply again, ending with ONLY a valid JSON object: "
                        '{"stage": "...", "constraint_updates": {...}, '
                        '"tool_call": null, "reasoning_note": "..."}'
                    )
                )
                delta, error, span = self._extract_json_delta(text)
                attempts += 1

            if delta is None:
                raise PlannerJSONDecodeError(
                    f"Planner failed to produce valid JSON after {attempts} attempts: {error}"
                )
            # Strip the JSON delta's own character span back out of the
            # text before it's surfaced to the user — see
            # _extract_json_delta's docstring for the bug this fixes.
            if span is not None:
                start, end = span
                conversational_text = (text[:start] + text[end:]).strip()
            else:
                conversational_text = text
            return {"text": conversational_text, "delta": delta, "attempts": attempts}

        result = await asyncio.to_thread(_run_sync)
        return {
            "tier": self.spec.tier.value,
            "reply_text": result["text"],
            "design_state_delta": result["delta"],
            "retrieved_critique_ids": [r.record_id for r in retrieved],
        }

    def _build_system_prompt(
        self,
        constraints: dict,
        retrieved: list,
        prior_critique: dict | None = None,
    ) -> str:
        intent_clause = ""
        if constraints and constraints.get("original_intent"):
            intent_clause = f"Original user goal: {constraints['original_intent']!r}. "

        system = (
            "You are Krisna's design planner. Discuss intent, refine "
            f"style/constraints, and describe the sketch to generate. "
            f"{intent_clause}"
            f"Current constraints: {constraints}. "
            "End every reply with a single JSON object on its own line matching "
            'this shape: {"stage": "conversing|sketching|finalizing|finalized|'
            'critiquing", "constraint_updates": {"...": "..."}, "tool_call": '
            '"<tool name or null>", "reasoning_note": "<=2 sentences"}'
        )
        if prior_critique:
            critique_lines = [f"Overall Score: {prior_critique.get('overall_score', 'N/A')}"]
            dims = prior_critique.get("dimensions", {})
            if isinstance(dims, dict):
                for dim_name, dim_val in dims.items():
                    note = dim_val.get("note", "") if isinstance(dim_val, dict) else str(dim_val)
                    if note:
                        critique_lines.append(f"- {dim_name}: {note}")
            edits = prior_critique.get("suggested_edits", [])
            if edits:
                critique_lines.append("Suggested Edits to address in this turn:")
                for edit in edits:
                    if isinstance(edit, dict):
                        instr = edit.get("instruction", "")
                        region = edit.get("region", "")
                        critique_lines.append(f"  * [{region}] {instr}" if region else f"  * {instr}")
                    else:
                        critique_lines.append(f"  * {edit}")
            system += "\n\nPrior Critic Feedback & Suggested Edits:\n" + "\n".join(critique_lines)

        if retrieved:
            # Real human UICrit critique snippets — this is the retrieval
            # context replacing the removed fine-tune. Each snippet is
            # already truncated to 400 chars at the data-forge export step
            # (see planner_rag.py's schema note), so no further truncation
            # needed here.
            examples = "\n".join(f"- {r.note}" for r in retrieved if r.note)
            if examples:
                system += (
                    "\n\nFor grounding, here is real design-critique feedback from "
                    f"human reviewers on related UI screens:\n{examples}"
                )
        return system

    @staticmethod
    def _extract_json_delta(text: str) -> tuple[dict | None, str | None, tuple[int, int] | None]:
        """Find the last well-formed {...} object in `text` and validate it
        has the minimum required design-state-delta keys. Returns
        (delta, None, (start, end)) on success or (None, error_message, None)
        on failure — never raises, so the retry loop above can decide what
        to do.

        Also returns the JSON object's character span within `text`. This
        matters: without it, the raw text (JSON object still attached)
        flowed unmodified all the way through to the chat UI, which
        renders it verbatim — every planner turn showed the user a reply
        with a raw JSON object glued onto the end. The caller strips this
        span out before surfacing the text.
        """
        try:
            end = text.rindex("}") + 1
            # Walk backward from the last '}' to find its matching '{',
            # handling nested objects (constraint_updates is itself a dict).
            depth = 0
            start = None
            for i in range(end - 1, -1, -1):
                if text[i] == "}":
                    depth += 1
                elif text[i] == "{":
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start is None:
                return None, "no balanced JSON object found", None
            parsed = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError) as e:
            return None, f"JSON parse error: {e}", None

        if not isinstance(parsed, dict):
            return None, "top-level value is not a JSON object", None
        if "stage" not in parsed:
            return None, "missing required key 'stage'", None
        valid_stages = {"conversing", "sketching", "finalizing", "finalized", "critiquing"}
        if parsed["stage"] not in valid_stages:
            return None, f"'stage' must be one of {sorted(valid_stages)}, got {parsed['stage']!r}", None

        parsed.setdefault("constraint_updates", {})
        parsed.setdefault("tool_call", None)
        parsed.setdefault("reasoning_note", "")
        return parsed, None, (start, end)
