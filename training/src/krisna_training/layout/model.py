"""Small continuous layout-flow model primitive.

This is a model building block, not a claim that Krisna has completed a
LayoutFlow training run. It predicts continuous box velocities and node-type
logits from padded layout tokens plus a conditioning mask.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayoutFlowConfig:
    max_nodes: int = 64
    type_count: int = 10
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1


def build_layout_flow_model(config: LayoutFlowConfig):
    import torch
    import torch.nn as nn

    class LayoutFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = config
            self.input = nn.Linear(4 + config.type_count + 1, config.hidden_dim)
            layer = nn.TransformerEncoderLayer(config.hidden_dim, config.n_heads, config.hidden_dim * 4,
                                                dropout=config.dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
            self.time = nn.Linear(1, config.hidden_dim)
            self.box_velocity = nn.Linear(config.hidden_dim, 4)
            self.type_logits = nn.Linear(config.hidden_dim, config.type_count)

        def forward(self, boxes, type_one_hot, condition_mask, timestep):
            x = torch.cat([boxes, type_one_hot, condition_mask.unsqueeze(-1)], dim=-1)
            x = self.input(x) + self.time(timestep.reshape(-1, 1, 1))
            x = self.encoder(x)
            return self.box_velocity(x), self.type_logits(x)

    return LayoutFlow()
