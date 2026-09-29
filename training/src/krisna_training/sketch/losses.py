"""Combined loss: masked-token cross-entropy (the generator objective) +
Token-Critic BCE (an auxiliary head trained to predict whether the
generator's own top-1 guess at each masked position is actually correct —
this is exactly the signal inference/maskgit_model.py's sampler multiplies
into its confidence gating at inference time, so it has to be trained
against ground truth, not just wired up and hoped to work).
"""

from __future__ import annotations


def compute_loss(
    logits, critic_scores, targets, mask, critic_loss_weight: float = 0.5,
    label_smoothing: float = 0.1,
):
    """logits: [B, N, vocab_size]. critic_scores: [B, N] in [0,1].
    targets: [B, N] ground-truth token ids (meaningful only where mask=1).
    mask: [B, N] {0,1} float/long — 1 at masked (loss-relevant) positions.

    label_smoothing defaults to 0.1, matching Chang et al. 2022's
    (MaskGIT) reported training setup — previously this was left at
    PyTorch's F.cross_entropy default of 0.0, an unintentional deviation
    caught during a design-sync review, not a considered choice.

    Returns (total_loss, token_loss, critic_loss) — all scalars.
    """
    import torch
    import torch.nn.functional as F

    mask_f = mask.float()
    num_masked = mask_f.sum().clamp(min=1.0)

    # Token cross-entropy, masked positions only.
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none", label_smoothing=label_smoothing,
    ).reshape(targets.shape)
    token_loss = (ce * mask_f).sum() / num_masked

    # Token-Critic target: 1.0 where the generator's own top-1 prediction
    # matches ground truth, 0.0 otherwise — computed from the SAME forward
    # pass's logits, detached (no gradient flows into the generator through
    # this target construction; only the critic head learns from it).
    with torch.no_grad():
        predicted = logits.argmax(dim=-1)
        critic_target = (predicted == targets).float()

    critic_bce = F.binary_cross_entropy(critic_scores, critic_target, reduction="none")
    critic_loss = (critic_bce * mask_f).sum() / num_masked

    total = token_loss + critic_loss_weight * critic_loss
    return total, token_loss.detach(), critic_loss.detach()
