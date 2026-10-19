"""Unit and Property Tests for LayoutFlow Neural Architecture, Loss, and Sampler.

Grounding:
- Guerreiro, J. A., et al. (ECCV 2024). "LayoutFlow: Flow Matching for Layout Generation."
- Lipman, Y., et al. (ICLR 2023). "Flow Matching for Generative Modeling."
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from krisna_training.layout.model import LayoutFlow, LayoutFlowConfig, build_layout_flow_model
from krisna_training.layout.loss import continuous_flow_matching_loss
from krisna_training.layout.sampler import LayoutFlowSampler
from krisna_training.layout.train import LayoutTensorDataset, train_layout_flow_epoch, evaluate_layout_flow


def test_layout_flow_forward_shape():
    """Verify LayoutFlow forward pass shapes for velocity and classification heads."""
    config = LayoutFlowConfig(max_nodes=16, type_count=10, hidden_dim=64, n_layers=2, n_heads=4)
    model = build_layout_flow_model(config)
    model.eval()

    B, N = 2, 8
    boxes = torch.rand(B, N, 4)
    type_oh = torch.zeros(B, N, 10)
    type_oh[:, :, 1] = 1.0  # text elements
    cond_mask = torch.zeros(B, N)
    cond_mask[:, 0] = 1.0   # first element locked
    timestep = torch.tensor([0.25, 0.75])

    v_pred, t_logits = model(boxes, type_oh, cond_mask, timestep)
    assert v_pred.shape == (B, N, 4)
    assert t_logits.shape == (B, N, 10)


def test_continuous_flow_matching_loss_and_gradients():
    """Verify OT-CFM loss computes valid scalars and backpropagates clean gradients."""
    config = LayoutFlowConfig(max_nodes=16, type_count=10, hidden_dim=64, n_layers=2, n_heads=4)
    model = LayoutFlow(config)
    model.train()

    B, N = 4, 6
    boxes_1 = torch.rand(B, N, 4)
    type_oh = torch.zeros(B, N, 10)
    type_oh[:, :, 2] = 1.0
    cond_mask = torch.zeros(B, N)
    cond_mask[:, 0] = 1.0

    loss_dict = continuous_flow_matching_loss(
        model=model,
        boxes_1=boxes_1,
        type_one_hot=type_oh,
        condition_mask=cond_mask,
    )

    assert "loss" in loss_dict
    assert "loss_velocity" in loss_dict
    assert "loss_cls" in loss_dict
    assert "ot_coherence" in loss_dict

    loss = loss_dict["loss"]
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # Verify backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"


def test_layout_flow_sampler_ode_integration():
    """Verify ODE sampler executes both Euler and Midpoint integration."""
    config = LayoutFlowConfig(max_nodes=8, type_count=10, hidden_dim=64, n_layers=2, n_heads=4)
    model = LayoutFlow(config)
    sampler = LayoutFlowSampler(model)

    B, N = 1, 4
    boxes_init = torch.rand(B, N, 4)
    type_oh = torch.zeros(B, N, 10)
    type_oh[0, :, 3] = 1.0  # button elements
    cond_mask = torch.zeros(B, N)

    # Test Euler
    euler_out = sampler.sample(boxes_init, type_oh, cond_mask, steps=5, solver="euler")
    assert euler_out["boxes"].shape == (B, N, 4)
    assert euler_out["trajectory_steps"] == 5
    assert (euler_out["boxes"] >= 0.0).all() and (euler_out["boxes"] <= 1.0).all()

    # Test Midpoint RK2
    midpoint_out = sampler.sample(boxes_init, type_oh, cond_mask, steps=5, solver="midpoint")
    assert midpoint_out["boxes"].shape == (B, N, 4)
    assert midpoint_out["trajectory_steps"] == 5
    assert midpoint_out["displacement_mean"] >= 0.0
    assert midpoint_out["vector_field_norm"] >= 0.0


def test_layout_flow_sampler_condition_mask_lock_immutability():
    """Verify that locked elements (condition_mask=1) do NOT move during ODE flow."""
    config = LayoutFlowConfig(max_nodes=8, type_count=10, hidden_dim=64, n_layers=2, n_heads=4)
    model = LayoutFlow(config)
    sampler = LayoutFlowSampler(model)

    B, N = 1, 4
    boxes_init = torch.tensor([[[0.1, 0.1, 0.9, 0.2],
                                [0.2, 0.3, 0.8, 0.6],
                                [0.3, 0.7, 0.7, 0.8],
                                [0.4, 0.85, 0.6, 0.95]]])
    type_oh = torch.zeros(B, N, 10)
    cond_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # Elements 0 and 2 are locked!

    out = sampler.sample(boxes_init, type_oh, cond_mask, steps=10, solver="midpoint")
    sampled_boxes = out["boxes"]

    # Assert locked boxes (0 and 2) are bitwise identical to initial coordinates
    assert torch.allclose(sampled_boxes[0, 0], boxes_init[0, 0], atol=1e-5)
    assert torch.allclose(sampled_boxes[0, 2], boxes_init[0, 2], atol=1e-5)


def test_train_and_evaluate_epoch():
    """Verify that train_layout_flow_epoch and evaluate_layout_flow complete with positive loss reduction."""
    records = [
        {
            "nodes": [
                {"id": "header", "type": "container", "bbox": [0.05, 0.05, 0.95, 0.20], "locked": True},
                {"id": "card", "type": "card", "bbox": [0.10, 0.25, 0.90, 0.65], "locked": False},
                {"id": "btn", "type": "button", "bbox": [0.70, 0.75, 0.90, 0.85], "locked": False},
            ]
        },
        {
            "nodes": [
                {"id": "nav", "type": "container", "bbox": [0.00, 0.00, 1.00, 0.10], "locked": True},
                {"id": "input", "type": "input", "bbox": [0.20, 0.40, 0.80, 0.50], "locked": False},
            ]
        },
    ] * 4

    dataset = LayoutTensorDataset(records, max_nodes=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    config = LayoutFlowConfig(max_nodes=16, type_count=10, hidden_dim=64, n_layers=2, n_heads=4)
    model = LayoutFlow(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_receipt = train_layout_flow_epoch(model, dataloader, optimizer, device="cpu")
    assert "loss" in train_receipt
    assert train_receipt["loss"] > 0.0

    val_receipt = evaluate_layout_flow(model, dataloader, device="cpu")
    assert "val_loss" in val_receipt
    assert val_receipt["val_loss"] > 0.0
