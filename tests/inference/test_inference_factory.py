from __future__ import annotations

import importlib
import inspect
import pytest

from krisna_inference.backends.factory import real_backend_factory
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


def test_factory_returns_correct_class_per_tier():
    from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
    from krisna_inference.backends.layout_backend import LayoutBackend
    from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend

    expected = {
        Tier.PLANNER: PlannerBackendLlama,
        Tier.SKETCH: LayoutBackend,
        Tier.POLISH_DEFAULT: QwenImage21Backend,
        Tier.POLISH_QUALITY: QwenImage21Backend,
    }
    for tier, cls in expected.items():
        backend = real_backend_factory(REGISTRY[tier])
        assert isinstance(backend, cls)
        assert backend.spec.tier == tier
        if tier in (Tier.POLISH_DEFAULT, Tier.POLISH_QUALITY):
            assert backend.strict is True
            assert backend.device == "cuda"
            assert backend.revision == "d26bb61231c349cf6b7896fa83353113880e1ba3"


class TestFrozenModelsHaveNoLoraWiring:
    """Regression guard for the freeze fix: Planner and Qwen Polish Quality
    must never accept a lora_adapter_path without a deliberate reviewed change.
    Polish Default is the one tier that accepts a verified DreamBooth LoRA.
    """

    def test_planner_backend_has_no_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.PLANNER])
        assert not hasattr(backend, "lora_adapter_path")

    def test_polish_quality_backend_has_no_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
        assert backend.lora_adapter_path is None

    def test_polish_quality_does_not_inherit_default_lora_env(self, monkeypatch):
        monkeypatch.setenv("KRISNA_POLISH_LORA_PATH", "should-not-load")
        backend = real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
        assert backend.lora_adapter_path is None

    def test_polish_default_backend_has_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.POLISH_DEFAULT])
        assert hasattr(backend, "lora_adapter_path")

    def test_factory_module_source_has_no_removed_env_vars(self):
        import krisna_inference.backends.factory as factory_module

        source = inspect.getsource(factory_module)
        for removed_env_var in (
            "KRISNA_PLANNER_LORA_PATH",
            "KRISNA_CRITIC_LORA_PATH",
        ):
            assert f'os.environ.get("{removed_env_var}"' not in source


class TestLowVramModeFactoryWiring:
    def test_low_vram_env_var_enables_offload_on_polish_default(self, monkeypatch):
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            backend = factory_module.real_backend_factory(REGISTRY[Tier.POLISH_DEFAULT])
            assert backend.enable_cpu_offload is True
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)

    def test_low_vram_env_var_enables_offload_on_polish_quality(self, monkeypatch):
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            backend = factory_module.real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
            assert backend.enable_cpu_offload is True
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)

    def test_default_mode_leaves_offload_disabled(self):
        polish_default = real_backend_factory(REGISTRY[Tier.POLISH_DEFAULT])
        assert polish_default.enable_cpu_offload is False
        backend = real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
        assert backend.enable_cpu_offload is False

    def test_planner_never_offloaded_even_in_low_vram_mode(self, monkeypatch):
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            planner = factory_module.real_backend_factory(REGISTRY[Tier.PLANNER])
            assert not getattr(planner, "enable_cpu_offload", False)
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)
