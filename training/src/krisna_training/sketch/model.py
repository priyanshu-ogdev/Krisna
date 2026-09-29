"""The concrete architecture behind inference/maskgit_model.py's sampling
contract: `.forward(tokens, mask, prompt_embedding) -> (logits, critic_scores)`.

A standard bidirectional (BERT/ViT-style, non-causal) transformer over VQ
token embeddings, with:
  - a learned MASK token embedding (id = vocab_size, one past the VQGAN
    codebook's own token ids)
  - EITHER learned absolute positional embeddings sized to the token grid
    (interpolatable via pos_embed.py for progressive-resolution training —
    the default, exact prior behavior, use_2d_rope=False) OR 2D axial
    rotary position embeddings (use_2d_rope=True — see SketchModelConfig
    and RotaryMultiheadSelfAttention below for the full rationale: this
    is the SOTA-aligned option, matching real, cited practice
    (Seedream 3.0's technical report, arXiv:2504.11346, names
    "cross-modality RoPE" as one of its four key pretraining
    techniques; FLUX and SD3-class DiTs use RoPE-family position
    encoding over learned absolute tables for the same reason: it
    generalizes to any grid size with NO retraining/interpolation,
    unlike a fixed-size learned table)
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
    vocab_size: int = 16384          # VQGAN codebook size (see vq_tokenizer.py's
                                       # module docstring — this is the plain
                                       # CompVis vqgan_imagenet_f16_16384
                                       # checkpoint's real, verified size)
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

    # UPGRADE (SOTA research pass, this review — see this module's
    # docstring for citations): 2D axial rotary position embeddings
    # instead of a learned absolute table. Defaults to False — an exact,
    # zero-behavior-change default, since flipping this changes the
    # model's real parameter count (no more `pos_embed` table) and is a
    # genuinely different architecture existing checkpoints can't be
    # loaded into (strict=True state_dict loading would fail on
    # pos_embed's absence/presence mismatch — this is a NEW-run opt-in,
    # not a drop-in upgrade for an already-trained checkpoint). See
    # RotaryMultiheadSelfAttention's docstring below for the full
    # rotation math and the specific, deliberate choice for how the
    # prepended prompt-conditioning token is positioned.
    use_2d_rope: bool = False
    rope_base: float = 10000.0       # standard RoPE base (Su et al., RoFormer,
                                       # arXiv:2104.09864) — same default every
                                       # major RoPE-using model (LLaMA, FLUX,
                                       # SD3-class DiTs) ships with; no
                                       # project-specific reason to deviate.

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
    with existing checkpoints exactly. The use_2d_rope=True path (see
    RotaryMultiheadSelfAttention below) reuses this exact same
    layer-by-layer checkpointing loop, just over a different layer stack.
    """
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

    class RotaryMultiheadSelfAttention(nn.Module):
        """2D axial RoPE self-attention (UPGRADE, SOTA research pass).

        WHY this exists instead of nn.MultiheadAttention: PyTorch's
        built-in attention has no hook for injecting a per-token rotation
        before the Q.K^T dot product, which is exactly what RoPE needs —
        so a fixed-size learned nn.Parameter pos_embed (this file's
        default path) is the alternative that DOES fit the built-in
        module, at the cost of needing pos_embed.py's bicubic
        interpolation hack whenever grid size changes between progressive-
        resolution stages. RoPE removes that hack entirely: rotation
        angles are a closed-form function of (row, col), defined for ANY
        grid size with no interpolation and no extra learned parameters.

        2D AXIAL SCHEME (real, cited convention — not invented here): for
        a token at grid position (row, col), split each attention head's
        head_dim in half. Apply standard 1D RoPE (Su et al. 2021,
        RoFormer, arXiv:2104.09864) using `row` as the position for the
        FIRST half of the head's dims, and `col` as the position for the
        SECOND half. This is the same axial-decomposition idea used by
        RoPE-ViT-family vision transformers and by RoPE-based DiTs
        (FLUX/SD3-class image-token position encoding, and Seedream 3.0's
        own "cross-modality RoPE" per its technical report,
        arXiv:2504.11346) — decomposing a 2D position into two
        independent 1D rotations is the standard way to extend RoPE
        beyond 1D sequences.

        THE PREPENDED PROMPT TOKEN (deliberate choice, stated explicitly
        rather than left implicit — this project's own established
        discipline): it is assigned grid position (row=0, col=0). This is
        not an arbitrary hack — RoPE's rotation at position 0 IS the
        identity (cos(0)=1, sin(0)=0 for every frequency), so this is
        mathematically equivalent to "the prompt token participates in
        attention with NO rotation applied," a clean, well-defined choice
        consistent with how many RoPE-based ViT/DiT implementations treat
        a prepended class/register/conditioning token.

        head_dim must be divisible by 4 (needs to split in half for
        row/col, then each half split again into rotation pairs) —
        asserted at construction; the default config (hidden_dim=512,
        n_heads=8 -> head_dim=64) satisfies this with room to spare.
        """

        def __init__(self, hidden_dim: int, n_heads: int, dropout: float, rope_base: float) -> None:
            super().__init__()
            assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            assert self.head_dim % 4 == 0, (
                f"head_dim={self.head_dim} must be divisible by 4 for 2D axial RoPE "
                "(split in half for row/col, then each half split into rotation pairs)"
            )
            self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = dropout
            self.rope_base = rope_base

            # Frequencies for ONE axis (row OR col), over head_dim // 4 pairs
            # — head_dim//2 dims per axis, in rotate-half pairs of 2.
            n_freqs = self.head_dim // 4
            freqs = 1.0 / (rope_base ** (torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs))
            self.register_buffer("freqs", freqs, persistent=False)

        def _rope_cos_sin(self, positions: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """positions: [N] long (row or col indices for N tokens).
            Returns (cos, sin), each [N, head_dim // 4], for use against
            ONE half-of-a-half of the head dim (see forward())."""
            angles = positions.float().unsqueeze(-1) * self.freqs.unsqueeze(0)  # [N, head_dim//4]
            return torch.cos(angles), torch.sin(angles)

        @staticmethod
        def _apply_rope(x_axis_half: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
            """x_axis_half: [..., N, d] where d = head_dim // 2 (one axis'
            share of the head dim). cos/sin: [N, d // 2]. Rotate-half
            convention (GPT-NeoX/LLaMA-style, split-half pairing rather
            than interleaved — the more common real implementation, and
            simpler to get right than interleaving)."""
            d = x_axis_half.shape[-1]
            x1, x2 = x_axis_half[..., : d // 2], x_axis_half[..., d // 2 :]
            # cos/sin broadcast over any leading (batch, head) dims via
            # unsqueeze — positions vary along the sequence dim (N),
            # which sits right before the feature dim in x1/x2.
            # DEFENSIVE CAST (real risk, flagged rather than left to
            # chance — this environment has no torch install to verify
            # numerically): this project runs its whole training loop
            # under torch.autocast(dtype=torch.bfloat16) (train.py), so
            # q/k here may arrive as bf16 while `freqs`/cos/sin are
            # fp32 buffers. Elementwise ops between mismatched float
            # dtypes are NOT reliably auto-promoted the way autocast
            # handles its own whitelisted ops (matmul/conv) — explicit
            # casting here is the same defensive pattern real RoPE
            # implementations (e.g. LLaMA's rotary embedding code) use,
            # rather than assuming autocast will paper over it.
            cos_b = cos.unsqueeze(0).unsqueeze(0).to(x_axis_half.dtype)  # [1, 1, N, d//2] vs x1 [B, H, N, d//2]
            sin_b = sin.unsqueeze(0).unsqueeze(0).to(x_axis_half.dtype)
            rotated1 = x1 * cos_b - x2 * sin_b
            rotated2 = x1 * sin_b + x2 * cos_b
            return torch.cat([rotated1, rotated2], dim=-1)

        def forward(self, x: "torch.Tensor", row_pos: "torch.Tensor", col_pos: "torch.Tensor") -> "torch.Tensor":
            """x: [B, N, hidden_dim]. row_pos/col_pos: [N] long, the 2D
            grid position of each of the N sequence entries (including
            the prepended prompt token at (0, 0) — see class docstring)."""
            B, N, _ = x.shape
            qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_heads, N, head_dim]

            half = self.head_dim // 2
            q_row, q_col = q[..., :half], q[..., half:]
            k_row, k_col = k[..., :half], k[..., half:]

            cos_row, sin_row = self._rope_cos_sin(row_pos)
            cos_col, sin_col = self._rope_cos_sin(col_pos)

            q = torch.cat(
                [self._apply_rope(q_row, cos_row, sin_row), self._apply_rope(q_col, cos_col, sin_col)], dim=-1
            )
            k = torch.cat(
                [self._apply_rope(k_row, cos_row, sin_row), self._apply_rope(k_col, cos_col, sin_col)], dim=-1
            )

            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )  # [B, n_heads, N, head_dim]
            attn_out = attn_out.transpose(1, 2).reshape(B, N, self.n_heads * self.head_dim)
            return self.out_proj(attn_out)

    class RoPEEncoderLayer(nn.Module):
        """Pre-LN transformer block using RotaryMultiheadSelfAttention —
        structurally equivalent to nn.TransformerEncoderLayer(norm_first=True)
        so gradient checkpointing (see _run_encoder below) works identically
        for both the RoPE and learned-pos_embed paths."""

        def __init__(self, cfg: SketchModelConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg.hidden_dim)
            self.attn = RotaryMultiheadSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout, cfg.rope_base)
            self.norm2 = nn.LayerNorm(cfg.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
            )
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x: "torch.Tensor", row_pos: "torch.Tensor", col_pos: "torch.Tensor") -> "torch.Tensor":
            x = x + self.dropout(self.attn(self.norm1(x), row_pos, col_pos))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x

    class SketchTransformer(nn.Module):
        def __init__(self, cfg: SketchModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.use_gradient_checkpointing = cfg.use_gradient_checkpointing
            self.use_2d_rope = cfg.use_2d_rope
            # +1 embedding row for the mask token, beyond the real vocab.
            self.token_embed = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_dim)
            self.prompt_proj = nn.Linear(cfg.prompt_dim, cfg.hidden_dim)

            if self.use_2d_rope:
                self.pos_embed = None
                self.layers = nn.ModuleList([RoPEEncoderLayer(cfg) for _ in range(cfg.n_layers)])
                # Row-major (row varies slower than col) — MUST match
                # vq_tokenizer.py's verified real token ordering (see that
                # module's decode() docstring: confirmed directly against
                # CompVis/taming-transformers' actual source this same
                # review pass), since these rotation angles are only
                # meaningful if they correspond to each token's REAL
                # spatial position in the image, not an arbitrary index.
                rows = torch.arange(cfg.grid_h).repeat_interleave(cfg.grid_w)
                cols = torch.arange(cfg.grid_w).repeat(cfg.grid_h)
                # Position (0, 0) prepended for the prompt-conditioning
                # token — see RotaryMultiheadSelfAttention's docstring for
                # why this is the correct, deliberate choice (identity
                # rotation), not an arbitrary filler value.
                self.register_buffer("_grid_rows", torch.cat([torch.zeros(1, dtype=torch.long), rows]), persistent=False)
                self.register_buffer("_grid_cols", torch.cat([torch.zeros(1, dtype=torch.long), cols]), persistent=False)
            else:
                self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_dim))
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
            if self.use_2d_rope:
                row_pos, col_pos = self._grid_rows, self._grid_cols
                if self.use_gradient_checkpointing and self.training:
                    for layer in self.layers:
                        x = torch.utils.checkpoint.checkpoint(
                            layer, x, row_pos, col_pos, use_reentrant=False
                        )
                    return x
                for layer in self.layers:
                    x = layer(x, row_pos, col_pos)
                return x

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

            x = self.token_embed(tokens)
            if not self.use_2d_rope:
                x = x + self.pos_embed.unsqueeze(0)
            prompt_ctx = self.prompt_proj(prompt_embedding).unsqueeze(1)  # [B, 1, hidden_dim]
            x = torch.cat([prompt_ctx, x], dim=1)  # prefix-conditioning

            x = self._run_encoder(x)
            x = self.final_norm(x)
            x = x[:, 1:, :]  # drop the prompt-context position before the heads

            logits = self.token_head(x)
            critic_scores = torch.sigmoid(self.critic_head(x)).squeeze(-1)
            return logits, critic_scores

    return SketchTransformer(config)
