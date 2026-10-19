"""Continuous Flow Matching (CFM) Layout Neural Network.

Grounding:
- Guerreiro, J. A., et al. (ECCV 2024). "LayoutFlow: Flow Matching for Layout Generation."
- Lipman, Y., et al. (ICLR 2023). "Flow Matching for Generative Modeling."
- Tan, Z., et al. (IEEE TPAMI 2025). "InstructLayout: Instruction-Driven Graphic Layout Generation via Semantic Graph Priors."

Predicts continuous bounding box velocities v_theta(b_t, t, c) and element type logits
conditioned on partial layout masks, locked nodes, and continuous time t in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LayoutFlowConfig:
    max_nodes: int = 64
    type_count: int = 10
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: shape (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class LayoutFlow(nn.Module):
    """Continuous Flow Matching Transformer for 2D Bounding Box Trajectories."""

    def __init__(self, config: LayoutFlowConfig | None = None):
        super().__init__()
        self.cfg = config or LayoutFlowConfig()
        in_dim = 4 + self.cfg.type_count + 1  # (x1, y1, x2, y2) + type_one_hot + condition_mask
        self.input_proj = nn.Linear(in_dim, self.cfg.hidden_dim)
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_dim,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.hidden_dim * 4,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(self.cfg.hidden_dim)

        # Output heads
        self.box_velocity = nn.Linear(self.cfg.hidden_dim, 4)
        self.type_logits = nn.Linear(self.cfg.hidden_dim, self.cfg.type_count)

    def forward(
        self,
        boxes: torch.Tensor,
        type_one_hot: torch.Tensor,
        condition_mask: torch.Tensor,
        timestep: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            boxes: Tensor of shape (B, N, 4) with bounding box coordinates in [0, 1].
            type_one_hot: Tensor of shape (B, N, type_count) with element category one-hot.
            condition_mask: Tensor of shape (B, N) where 1 indicates conditioned/locked nodes, 0 mutable.
            timestep: Tensor of shape (B,) or (B, 1) with continuous time t in [0, 1].
            padding_mask: Optional boolean Tensor of shape (B, N) where True indicates padded dummy nodes.

        Returns:
            Tuple of:
            - v_pred: Predicted box velocity tensor of shape (B, N, 4).
            - type_logits: Predicted category logits tensor of shape (B, N, type_count).
        """
        # Feature concatenation: (B, N, 4 + type_count + 1)
        if condition_mask.dim() == 2:
            cond_expanded = condition_mask.unsqueeze(-1)
        else:
            cond_expanded = condition_mask

        x = torch.cat([boxes, type_one_hot, cond_expanded], dim=-1)
        h = self.input_proj(x)  # (B, N, hidden_dim)

        # Time modulation
        t_emb = self.time_embed(timestep)  # (B, hidden_dim)
        h = h + t_emb.unsqueeze(1)  # Broadcast across sequence length N

        # Transformer encoder
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.final_norm(h)

        v_pred = self.box_velocity(h)
        t_logits = self.type_logits(h)
        return v_pred, t_logits


def build_layout_flow_model(config: LayoutFlowConfig | None = None) -> LayoutFlow:
    """Factory helper preserving backward compatibility."""
    return LayoutFlow(config or LayoutFlowConfig())
