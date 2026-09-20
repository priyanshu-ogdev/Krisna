from __future__ import annotations

import pytest

from krisna_inference.backends.factory import real_backend_factory
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


def test_factory_returns_correct_class_per_tier():
    from krisna_inference.backends.critic_backend import CriticBackend
    from krisna_inference.backends.planner_backend import PlannerBackend
    from krisna_inference.backends.polish_default_backend import ZImageTurboBackend
    from krisna_inference.backends.polish_quality_backend import QwenImageEditBackend
    from krisna_inference.backends.sketch_backend import SketchBackend

    expected = {
        Tier.PLANNER: PlannerBackend,
        Tier.SKETCH: SketchBackend,
        Tier.POLISH_DEFAULT: ZImageTurboBackend,
        Tier.POLISH_QUALITY: QwenImageEditBackend,
        Tier.CRITIC: CriticBackend,
    }
    for tier, cls in expected.items():
        backend = real_backend_factory(REGISTRY[tier])
        assert isinstance(backend, cls)
        assert backend.spec.tier == tier


@pytest.mark.asyncio
async def test_sketch_backend_load_fails_clearly_without_checkpoint():
    from krisna_inference.orchestrator.exceptions import BackendLoadError
    from krisna_inference.backends.sketch_backend import SketchBackend

    backend = SketchBackend(REGISTRY[Tier.SKETCH], checkpoint_path=None)
    with pytest.raises(BackendLoadError, match="no checkpoint_path configured"):
        await backend.load()


@pytest.mark.asyncio
async def test_sketch_backend_load_fails_clearly_on_missing_file(tmp_path):
    from krisna_inference.orchestrator.exceptions import BackendLoadError
    from krisna_inference.backends.sketch_backend import SketchBackend

    backend = SketchBackend(REGISTRY[Tier.SKETCH], checkpoint_path=str(tmp_path / "nope.pt"))
    with pytest.raises(BackendLoadError, match="not found"):
        await backend.load()


@pytest.mark.asyncio
async def test_critic_backend_load_fails_clearly_without_venv(tmp_path):
    from krisna_inference.orchestrator.exceptions import BackendLoadError
    from krisna_inference.backends.critic_backend import CriticBackend

    backend = CriticBackend(REGISTRY[Tier.CRITIC], worker_python=str(tmp_path / "does-not-exist"))
    with pytest.raises(BackendLoadError, match="isolated venv"):
        await backend.load()


class TestFrozenModelsHaveNoLoraWiring:
    """Regression guard for the freeze fix: Planner, Critic, and Qwen-
    Image-Edit-2511 (POLISH_QUALITY) must never accept a lora_adapter_path
    again without a deliberate, reviewed change — see factory.py's inline
    comments for why each was removed. Z-Image-Turbo (POLISH_DEFAULT) is
    the one exception; it's the actual fine-tuned renderer.
    """

    def test_planner_backend_has_no_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.PLANNER])
        assert not hasattr(backend, "lora_adapter_path")

    def test_critic_backend_has_no_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.CRITIC])
        assert not hasattr(backend, "lora_adapter_path")

    def test_polish_quality_backend_has_no_lora_param(self):
        backend = real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
        assert not hasattr(backend, "lora_adapter_path")

    def test_polish_default_backend_still_has_lora_param(self):
        """The one tier that SHOULD still support it — Z-Image-Turbo is
        the only renderer this project actually fine-tunes."""
        backend = real_backend_factory(REGISTRY[Tier.POLISH_DEFAULT])
        assert hasattr(backend, "lora_adapter_path")

    def test_factory_module_source_has_no_removed_env_vars(self):
        """Belt-and-suspenders: check the actual source text, not just
        instance attributes, in case a future edit adds the attribute
        back under a different code path than __init__."""
        import inspect

        import krisna_inference.backends.factory as factory_module

        source = inspect.getsource(factory_module)
        for removed_env_var in (
            "KRISNA_PLANNER_LORA_PATH",
            "KRISNA_CRITIC_LORA_PATH",
            "KRISNA_POLISH_QUALITY_LORA_PATH",
        ):
            # Allowed to appear in a comment explaining the removal, but
            # never as a live os.environ.get(...) call.
            assert f'os.environ.get("{removed_env_var}"' not in source, (
                f"{removed_env_var} should not be actively read — this tier is frozen"
            )


class TestLowVramModeFactoryWiring:
    def test_low_vram_env_var_enables_offload_on_polish_default(self, monkeypatch):
        """Polish Default used to be excluded from offload wiring on the
        (incorrect) assumption that its 8.0GB declared vram_gb — which
        assumed NF4 quantization the backend never applies — already fit
        a low-VRAM target. Corrected: this tier is bf16 (~14GB real), so
        it needs the same offload wiring Polish Quality already had. See
        model_registry.py's LOW_VRAM_REGISTRY entry and
        docs/review/13_ram_offload_and_precision_audit.md."""
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import importlib

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
        import importlib

        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            backend = factory_module.real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
            assert backend.enable_cpu_offload is True
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)

    def test_low_vram_env_var_sets_critic_max_gpu_gb(self, monkeypatch):
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import importlib

        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            backend = factory_module.real_backend_factory(REGISTRY[Tier.CRITIC])
            assert backend.max_gpu_gb == 11.5
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)

    def test_default_mode_leaves_offload_disabled(self):
        polish_default = real_backend_factory(REGISTRY[Tier.POLISH_DEFAULT])
        assert polish_default.enable_cpu_offload is False
        backend = real_backend_factory(REGISTRY[Tier.POLISH_QUALITY])
        assert backend.enable_cpu_offload is False
        critic = real_backend_factory(REGISTRY[Tier.CRITIC])
        assert critic.max_gpu_gb is None

    def test_planner_never_offloaded_even_in_low_vram_mode(self, monkeypatch):
        """Confirmed by design (see model_registry.py's LOW_VRAM_REGISTRY
        comment): Planner and Sketch are already small enough that
        offload buys nothing but latency. Polish Default is NOT in this
        group any more — see test_low_vram_env_var_enables_offload_on_polish_default
        above; it used to be, incorrectly."""
        monkeypatch.setenv("KRISNA_LOW_VRAM_MODE", "1")
        import importlib

        import krisna_inference.backends.factory as factory_module
        importlib.reload(factory_module)
        try:
            planner = factory_module.real_backend_factory(REGISTRY[Tier.PLANNER])
            assert planner.max_gpu_gb is None
        finally:
            monkeypatch.delenv("KRISNA_LOW_VRAM_MODE", raising=False)
            importlib.reload(factory_module)
