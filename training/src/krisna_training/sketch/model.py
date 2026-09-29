"""The concrete architecture behind inference/maskgit_model.py's sampling
contract: `.forward(tokens, mask, prompt_embedding) -> (logits, critic_scores)`.

A standard bidirectional (BERT/ViT-style, non-causal) transformer over VQ
token embeddings, with:
  - a learned MASK token embedding (id = vocab_size, one past the VQGAN
    codebook's own token ids)
  - learned positional embeddings sized to the token grid (interpolatable
    via pos_embed.py for progressive-resolution training)
  - prefix conditioning: the planner's prompt embedding is projected into
    model space and prepended as one extra sequence position, letting
    every token attend to it via ordinary self-attention — the simplest
    correct way to condition a bidirectional transformer without inventing
    a cross-attention module
  - two output heads: `token_head` (next-token-style logits over the real
    vocab, used at masked positions) and `critic_head` (a scalar
    "how much do I trust this prediction" logit per position — the
    Token-Critic signal `inference/maskgit_model.py`'s sampler multiplies
    into its confidence gating)

Sizing here targets §6.1's own scale precedent ("small-scale, ImageNet-
lineage precedent: ~2M images, 256->512px progressive training") — this is
a genuinely small transformer (default: 12 layers, 512 dim, ~44M params),
not an attempt at a large foundation model, consistent with §3's Non-goal
("No from-scratch foundation-model pretraining").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SketchModelConfig:
    vocab_size: int = 16384          # VQGAN codebook size (boris/vqgan_f16_16384 default)
    grid_h: int = 16
    grid_w: int = 16
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    prompt_dim: int = 768            # CLIP ViT-L/14 text-embedding dim — was
                                       # 4096 (an arbitrary placeholder from
                                       # before Phase 17's real CLIP
                                       # integration), left stale even after
                                       # both real training configs
                                       # (sketch_train_stage{1,2}_*.yaml)
                                       # correctly moved to 768. Corrected so
                                       # the default itself reflects what's
                                       # actually used, not a leftover guess.
    use_gradient_checkpointing: bool = False   # see build_model()'s docstring

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size  # one past the real codebook

    @property
    def seq_len(self) -> int:
        return self.grid_h * self.grid_w


def build_model(config: SketchModelConfig):
    """Returns a torch.nn.Module. Kept as a function (not a class users
    instantiate directly) so torch is only imported when actually building
    a model, preserving this package's lazy-import discipline.

    UPGRADE — `use_gradient_checkpointing`: Stage 2 (512px, grid 32x32 =
    1024 tokens/image, vs. Stage 1's 256) doesn't just need 4x the memory
    Stage 1 used — full self-attention is O(n^2) in sequence length, so
    attention FLOPs alone are ~16x more per sample; sketch_train_stage2_512.yaml's
    batch_size=16 (vs. Stage 1's 32) only offsets this to a net ~8x
    compute/memory increase per step. `nn.TransformerEncoderLayer` below
    (batch_first=True, norm_first=True, no explicit attention mask) likely
    dispatches to PyTorch's scaled_dot_product_attention fast path
    (memory-linear flash-attention, head_dim=64 in range) even during
    training, but that depends on PyTorch version/kernel selection this
    environment can't verify. Gradient checkpointing is the version-
    independent safeguard: recompute each encoder layer during backward
    instead of storing activations, trading ~30% more compute time for a
    hard reduction in peak memory. Implemented by manually iterating
    `self.encoder.layers` (see forward() below) rather than restructuring
    the module — `nn.TransformerEncoder` here has `norm=None` (final_norm
    is applied separately), so this preserves state_dict key compatibility
    with existing checkpoints exactly.
    """
    import torch
    import torch.nn as nn
    import torch.utils.checkpoint

    class SketchTransformer(nn.Module):
        def __init__(self, cfg: SketchModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.use_gradient_checkpointing = cfg.use_gradient_checkpointing
            # +1 embedding row for the mask token, beyond the real vocab.
            self.token_embed = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_dim)
            self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.prompt_proj = nn.Linear(cfg.prompt_dim, cfg.hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
            )
            self.final_norm = nn.LayerNorm(cfg.hidden_dim)

            self.token_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
            self.critic_head = nn.Linear(cfg.hidden_dim, 1)

        def _run_encoder(self, x: "torch.Tensor") -> "torch.Tensor":
            if self.use_gradient_checkpointing and self.training:
                for layer in self.encoder.layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                return x
            return self.encoder(x)

        def forward(self, tokens: "torch.Tensor", mask: "torch.Tensor", prompt_embedding: "torch.Tensor"):
            """tokens: [B, N] long, mask token id at masked positions.
            mask: [B, N] {0,1}, unused by the forward math itself (tokens
            already encodes maskedness via mask_token_id) but accepted to
            match the sampler's call signature and kept available for
            future masked-attention variants.
            prompt_embedding: [B, prompt_dim].
            Returns (logits [B, N, vocab_size], critic_scores [B, N] in [0,1]).
            """
            B, N = tokens.shape
            assert N == self.cfg.seq_len, f"expected seq_len={self.cfg.seq_len}, got {N}"

            x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
            prompt_ctx = self.prompt_proj(prompt_embedding).unsqueeze(1)  # [B, 1, hidden_dim]
            x = torch.cat([prompt_ctx, x], dim=1)  # prefix-conditioning

            x = self._run_encoder(x)
            x = self.final_norm(x)
            x = x[:, 1:, :]  # drop the prompt-context position before the heads

            logits = self.token_head(x)
            critic_scores = torch.sigmoid(self.critic_head(x)).squeeze(-1)
            return logits, critic_scores

    return SketchTransformer(config)
