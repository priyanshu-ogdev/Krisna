"""Tests for ModelEngine memory leak prevention during loading and unloading."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from data_forge.config import PipelineConfig, ModelSpec, EncoderSpec
from data_forge.inference.engine import ModelEngine


@pytest.mark.asyncio
async def test_stop_vllm_handles_dead_child_and_cleans_memory():
    """Verify that stop_vllm terminates the process tree without crashing on dead children,
    closes log handles, and calls gc and cuda empty_cache."""
    engine = ModelEngine()
    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    engine._vllm_process = fake_proc
    engine._current_model = "tier1"

    fake_log = MagicMock()
    engine._vllm_log_handle = fake_log

    with patch("psutil.Process") as mock_parent_cls, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("asyncio.sleep", return_value=None):

        mock_parent = MagicMock()
        child1 = MagicMock()
        child2 = MagicMock()

        # Simulate child1 being killed, child2 already terminated (NoSuchProcess)
        import psutil
        child2.kill.side_effect = psutil.NoSuchProcess(pid=99999)
        mock_parent.children.return_value = [child1, child2]
        mock_parent_cls.return_value = mock_parent

        await engine.stop_vllm()

        # Both children attempted, parent killed
        child1.kill.assert_called_once()
        child2.kill.assert_called_once()
        mock_parent.kill.assert_called_once()

        # Process and log handle cleaned up
        assert engine._vllm_process is None
        assert engine._current_model is None
        fake_log.close.assert_called_once()
        assert engine._vllm_log_handle is None

        # CUDA empty_cache invoked
        mock_empty_cache.assert_called_once()


def test_unload_clip_evacuates_to_cpu_and_clears_cache():
    """Verify that unload_clip moves the model to CPU before deletion and empties CUDA cache."""
    engine = ModelEngine()
    mock_model = MagicMock()
    mock_proc = MagicMock()
    engine._clip_model = mock_model
    engine._clip_processor = mock_proc

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty_cache:

        engine.unload_clip()

        mock_model.to.assert_called_once_with("cpu")
        assert engine._clip_model is None
        assert engine._clip_processor is None
        mock_empty_cache.assert_called_once()


def test_unload_encoders_evacuates_to_cpu_and_clears_cache():
    """Verify that unload_encoders moves encoder models to CPU before deletion and empties CUDA cache."""
    engine = ModelEngine()
    mock_vae = MagicMock()
    engine._encoders["z_image_vae"] = mock_vae

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty_cache:

        engine.unload_encoders()

        mock_vae.to.assert_called_once_with("cpu")
        assert len(engine._encoders) == 0
        mock_empty_cache.assert_called_once()


@pytest.mark.asyncio
async def test_session_context_managers_cleanup_on_exception():
    """Verify that session context managers trigger unload/stop even if an exception occurs."""
    config = PipelineConfig()

    # CLIP session cleanup on error
    with patch.object(ModelEngine, "load_clip"), \
         patch.object(ModelEngine, "unload_clip") as mock_unload_clip:
        with pytest.raises(RuntimeError, match="Simulated worker crash"):
            async with ModelEngine.clip_session(config):
                raise RuntimeError("Simulated worker crash")
        mock_unload_clip.assert_called_once()

    # Encoder session cleanup on error
    with patch.object(ModelEngine, "load_encoders"), \
         patch.object(ModelEngine, "unload_encoders") as mock_unload_encoders:
        with pytest.raises(RuntimeError, match="Simulated worker crash"):
            async with ModelEngine.encoder_session(config):
                raise RuntimeError("Simulated worker crash")
        mock_unload_encoders.assert_called_once()

    # vLLM session cleanup on error
    with patch.object(ModelEngine, "start_vllm"), \
         patch.object(ModelEngine, "stop_vllm") as mock_stop_vllm:
        with pytest.raises(RuntimeError, match="Simulated worker crash"):
            async with ModelEngine.vllm_session(config, "tier1"):
                raise RuntimeError("Simulated worker crash")
        mock_stop_vllm.assert_called_once()
