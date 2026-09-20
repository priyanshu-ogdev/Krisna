from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from krisna_training.planner.lora_config import (
    discover_target_modules,
    list_linear_module_names,
)


class FakeAttnBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)  # not a Linear — should never be matched


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 8)
        self.layers = nn.ModuleList([FakeAttnBlock() for _ in range(2)])
        self.lm_head = nn.Linear(8, 100)  # a Linear that WON'T match the standard patterns


def test_list_linear_module_names_finds_every_linear():
    model = FakeModel()
    names = list_linear_module_names(model)
    # 7 Linear per block * 2 blocks + lm_head
    assert len(names) == 15
    assert any(n.endswith("q_proj") for n in names)
    assert any(n == "lm_head" for n in names)


def test_discover_target_modules_matches_standard_patterns():
    model = FakeModel()
    matched = discover_target_modules(model)
    assert set(matched) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }
    assert "lm_head" not in matched
    assert "layer_norm" not in matched  # not even a Linear


def test_discover_target_modules_raises_when_nothing_matches():
    model = FakeModel()
    with pytest.raises(ValueError, match="No Linear modules matched"):
        discover_target_modules(model, include_patterns=("totally_nonexistent_proj",))


def test_discover_target_modules_custom_patterns():
    model = FakeModel()
    matched = discover_target_modules(model, include_patterns=("q_proj", "lm_head"))
    assert set(matched) == {"q_proj", "lm_head"}
