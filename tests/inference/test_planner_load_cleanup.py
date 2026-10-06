"""Regression test for a bug found on re-review
(docs/review/31_planner_retrieval_upgrade.md): PlannerBackend.load()'s
RAG-embedding step is a SECOND failure-prone step, downstream of the
model/tokenizer load already succeeding. swap_orchestrator.py's
_load_one only rolls back its own ledger bookkeeping on a raised
load() — it never calls the backend's own unload() — so without this
backend cleaning up after itself, a RAG-embedding load failure would
leak the already-resident model/tokenizer while the ledger incorrectly
believes that budget is free again.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krisna_inference.backends.planner_backend import PlannerBackend
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


def _fake_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = MagicMock()
    return tok


@pytest.mark.asyncio
async def test_rag_embedding_failure_cleans_up_already_loaded_model(monkeypatch, tmp_path):
    backend = PlannerBackend(
        REGISTRY[Tier.PLANNER],
        quantize=False,  # skip bitsandbytes entirely — irrelevant to what's under test
        rag_corpus_dir=str(tmp_path),  # any dir — corpus file need not exist for this test
    )
    monkeypatch.setenv("KRISNA_PLANNER_RAG_EMBEDDINGS", "1")

    fake_model = MagicMock()
    fake_model.eval.return_value = None

    def fail_embed_resolution():
        raise RuntimeError("simulated: sentence-transformers not installed in this build")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
    ), patch(
        "krisna_inference.backends.planner_rag.resolve_embed_fn_from_env",
        side_effect=fail_embed_resolution,
    ):
        with pytest.raises(BackendLoadError, match="RAG embedding model failed to load"):
            await backend.load()

    # The whole point of this test: the model/tokenizer that DID
    # successfully load must not be left dangling once load() has
    # raised — matching planner_backend_vllm.py's equivalent cleanup
    # (await self._kill()) on the same failure path.
    assert backend._model is None
    assert backend._tokenizer is None
    assert backend.is_loaded is False


@pytest.mark.asyncio
async def test_successful_load_without_rag_embeddings_is_unaffected(monkeypatch, tmp_path):
    """Sanity check: the fix above must not change the ordinary,
    embeddings-disabled (default) path at all."""
    backend = PlannerBackend(
        REGISTRY[Tier.PLANNER],
        quantize=False,
        rag_corpus_dir=str(tmp_path),
    )
    monkeypatch.delenv("KRISNA_PLANNER_RAG_EMBEDDINGS", raising=False)

    fake_model = MagicMock()
    fake_model.eval.return_value = None

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()
    ):
        await backend.load()

    assert backend.is_loaded is True
    assert backend._model is fake_model
