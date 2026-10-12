"""Sole consumer-release Planner runtime: llama.cpp + GGUF."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from krisna_inference.backends.planner_rag import UICritRAGIndex
from krisna_inference.backends.planner_schema import PlannerResponse
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend


class PlannerBackendLlama(ModelBackend):
    runtime_name = "llama.cpp"

    def __init__(self, spec, *, rag_corpus_dir: str | None = None):
        super().__init__(spec)
        self.model_path = os.environ.get("KRISNA_PLANNER_GGUF")
        self.mmproj_path = os.environ.get("KRISNA_PLANNER_MMPROJ")
        self.rag_corpus_dir = rag_corpus_dir
        self._llama = None
        self._rag_index = UICritRAGIndex()

    async def load(self) -> None:
        if not self.model_path or not Path(self.model_path).exists():
            raise BackendLoadError("KRISNA_PLANNER_GGUF must point to models/planner/*.gguf")

        def _load():
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise BackendLoadError("Install llama-cpp-python for the consumer Planner") from exc
            kwargs = {"model_path": self.model_path, "n_ctx": 16384, "n_gpu_layers": -1, "verbose": False}
            if self.mmproj_path:
                kwargs["clip_model_path"] = self.mmproj_path
            self._llama = Llama(**kwargs)

        await asyncio.to_thread(_load)
        if self.rag_corpus_dir:
            corpus = Path(self.rag_corpus_dir) / "planner_rag_corpus/uicrit_critiques.jsonl"
            self._rag_index = await asyncio.to_thread(UICritRAGIndex.from_jsonl, corpus)
        self._loaded = True

    async def unload(self) -> None:
        self._llama = None
        self._rag_index = UICritRAGIndex()
        self._loaded = False

    def _prompt(self, message, constraints, history, critique, review):
        retrieved = self._rag_index.retrieve(message, k=3)
        system = (
            "You are Krisna's frozen design Planner. Return one JSON object with "
            "schema_version, reply_text, design_state_delta, retrieved_critique_ids, "
            "and runtime_metadata. "
            "Never invent unsupported tool calls. Current constraints: " + str(constraints)
        )
        if review:
            system += " Review the current design, identify concrete defects, and propose constraint updates."
        if critique:
            system += " Prior review: " + str(critique)
        if retrieved:
            system += " Human UICrit grounding: " + " | ".join(r.note for r in retrieved)
        messages = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            role = "assistant" if turn.get("role") == "planner" else turn.get("role", "user")
            if turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": message or "Continue the design."})
        return messages, retrieved

    async def run(self, message="", constraints=None, conversation_history=None, prior_critique=None, review=False, **kwargs):
        if not self._loaded:
            raise RuntimeError("Planner backend not loaded")
        messages, retrieved = self._prompt(message, constraints or {}, conversation_history, prior_critique, review)

        def _generate():
            result = self._llama.create_chat_completion(
                messages=messages, max_tokens=512, temperature=0.7,
                response_format={"type": "json_object"},
            )
            text = result["choices"][0]["message"]["content"]
            import json
            parsed = json.loads(text)
            response = PlannerResponse.model_validate(parsed)
            return {"text": response.reply_text, "delta": response.design_state_delta.model_dump()}

        result = await asyncio.to_thread(_generate)
        return {
            "tier": self.spec.tier.value,
            "reply_text": result["text"],
            "design_state_delta": result["delta"],
            "retrieved_critique_ids": [r.record_id for r in retrieved],
            "runtime_metadata": {"runtime": self.runtime_name, "model_id": self.model_path},
        }
