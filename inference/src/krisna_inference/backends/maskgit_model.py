"""UI-domain masked-generative transformer — MaskGIT/MaskGIL lineage.

PRD §6.1: "no public T2I checkpoint exists in the MaskGIT/MaskGIL/Halton-
scheduler lineage as of the last check ... tracked as its own research
effort with independent timeline risk." There is genuinely no pretrained
checkpoint to load here — this module is the architecture + inference-time
sampler that a future training run would produce weights for, built now so
SketchBackend has something real to load once that checkpoint exists,
rather than blocking the whole inference layer on training being finished.

Implements the three mechanisms named in §5's architecture diagram:
  - Cosine mask schedule (fraction of tokens revealed per round)
  - Halton-sequence token selection (spatially dispersed, not the raw
    confidence top-k MaskGIT originally used — Halton avoids clustering
    revealed tokens in one image region early on)
  - Token-Critic-style confidence gating (a small second head scores
    "is this predicted token trustworthy", separate from the generator's
    own softmax confidence, before a token is committed)
"""

from __future__ import annotations

import math


def cosine_mask_schedule(step: int, total_steps: int) -> float:
    """Fraction of tokens that should be MASKED (not yet revealed) after
    this step, per MaskGIT's cosine schedule. step is 1-indexed."""
    ratio = step / total_steps
    return math.cos(ratio * math.pi / 2)


def halton_sequence(n: int, base: int) -> list[float]:
    """First n terms of the base-`base` Halton sequence, in [0, 1)."""
    seq = []
    for i in range(1, n + 1):
        f, r, idx = 1.0, 0.0, i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq.append(r)
    return seq


def halton_token_order(grid_h: int, grid_w: int) -> list[int]:
    """A spatially-dispersed visiting order over an H×W token grid, built
    from a 2D Halton sequence (bases 2 and 3) mapped to grid coordinates.
    Early positions in the returned order are maximally spread out rather
    than clustered — this is what keeps early-round sketch reveals from
    committing to (say) just the top-left corner first."""
    n = grid_h * grid_w
    xs = halton_sequence(n, 2)
    ys = halton_sequence(n, 3)
    coords = [(int(x * grid_w) % grid_w, int(y * grid_h) % grid_h) for x, y in zip(xs, ys)]
    seen = set()
    order = []
    for x, y in coords:
        idx = y * grid_w + x
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    # Fallback: any grid cell the Halton sequence didn't hit due to
    # collisions gets appended in raster order at the end.
    for idx in range(n):
        if idx not in seen:
            order.append(idx)
    return order


class MaskGITSketchModel:
    """Wraps a checkpoint produced by training.sketch's training loop
    (architecture defined in training.sketch.model.build_model — see
    checkpoint_io.py for why the checkpoint stores a state_dict + config
    rather than a pickled model object) plus the inference-time sampling
    loop described in the module docstring.

    The *sampling contract* below is what actually matters for anything
    calling this class: given a loaded module exposing `.forward(tokens,
    mask, prompt_embedding) -> (logits, critic_scores)`, `sample()` runs
    the cosine-schedule + Halton-order + Token-Critic-gated iterative
    decode described in §5. `from_checkpoint()` is currently the only way
    to get such a module, and it's tied to training.sketch.model's
    specific architecture — that's a real coupling (not the fully
    decoupled "any architecture that matches the contract" framing an
    earlier draft of this docstring implied), traded off deliberately to
    avoid the fragility of pickling live nn.Module objects across
    training/inference process boundaries.
    """

    def __init__(self, torch_module, grid_h: int, grid_w: int, mask_token_id: int) -> None:
        self.module = torch_module  # a torch.nn.Module with the forward() contract above
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.mask_token_id = mask_token_id

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda") -> "MaskGITSketchModel":
        import torch

        from krisna_training.sketch.model import build_model

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Rebuilt from config + state_dict, not unpickled as a live object —
        # see training/sketch/checkpoint_io.py's docstring for why (pickling
        # nn.Module instances directly is fragile, and build_model()'s
        # SketchTransformer is deliberately a function-local class to
        # preserve this package's lazy-torch-import discipline, which plain
        # pickling can't handle at all). This is also what keeps the
        # architecture definition in exactly one place — model.py — instead
        # of duplicating it on the inference side.
        module = build_model(ckpt["config"])
        module.load_state_dict(ckpt["model_state_dict"])
        module.to(device).eval()
        return cls(
            torch_module=module,
            grid_h=ckpt["grid_h"],
            grid_w=ckpt["grid_w"],
            mask_token_id=ckpt["mask_token_id"],
        )

    def sample(
        self,
        prompt_embedding,
        num_rounds: int = 8,
        prior_tokens: "list[int] | None" = None,
        guidance_scale: float = 0.0,
        uncond_embedding=None,
    ) -> dict:
        """Runs one interactive sketch round (~8 steps per §5's architecture
        diagram — cheap enough for many conversational rounds).

        guidance_scale: classifier-free guidance strength, t, per Muse's
        own formula (Chang et al. 2023, "Muse: Text-To-Image Generation
        via Masked Generative Transformers", §2.7 — the actual T2I
        MaskGIT-lineage paper this project's architecture already
        follows, whose 10% training-time conditioning-dropout rate
        matches this project's own `cfg_dropout_prob` default, see
        sketch/train.py):

            l_g = (1 + t) * l_c - t * l_u

        where l_c is the conditional logits (this prompt_embedding) and
        l_u is the unconditional logits (uncond_embedding, zeros by
        default — the same zero-vector convention training's collate_fn
        uses for CFG-dropped samples). t=0 recovers plain conditional
        decoding with no extra cost (unconditional forward pass skipped
        entirely) — the default, so existing callers unaffected.

        Per Muse's own reported refinement: guidance is linearly ramped
        from 0 up to `guidance_scale` across sampling rounds rather than
        held constant, to preserve sample diversity in early rounds
        while sharpening prompt adherence by the final rounds.
        Implemented as `t_step = guidance_scale * (step / num_rounds)`.
        See docs/review/17_sketch_inference_conditioning_and_cfg.md.

        Returns {"tokens": [...], "confidence_map": [...]} — a full H*W
        grid where masked positions from THIS round are filled with the
        model's best guess, low-confidence ones stay flagged in
        confidence_map for the next round to reconsider.
        """
        import torch

        n = self.grid_h * self.grid_w
        tokens = list(prior_tokens) if prior_tokens else [self.mask_token_id] * n
        visit_order = halton_token_order(self.grid_h, self.grid_w)
        confidence = [0.0] * n

        device = next(self.module.parameters()).device
        if guidance_scale > 0.0 and uncond_embedding is None:
            uncond_embedding = torch.zeros_like(prompt_embedding)

        for step in range(1, num_rounds + 1):
            mask_fraction = cosine_mask_schedule(step, num_rounds)
            # Target CUMULATIVE reveal count by the end of this step — NOT
            # a per-step increment (that was the bug: dividing by
            # num_rounds again after cosine_mask_schedule already encodes
            # the full schedule double-counts it, and rounding down on
            # every step compounds into leftover masked positions that
            # never get revealed by the final round). On the final round,
            # force full reveal regardless of the schedule's rounding —
            # sample() must never return a grid with mask_token_id still
            # in it.
            target_revealed = n if step == num_rounds else round(n * (1.0 - mask_fraction))
            current_revealed = sum(1 for t in tokens if t != self.mask_token_id)
            n_reveal_this_round = max(0, target_revealed - current_revealed)
            if n_reveal_this_round == 0 and current_revealed < n:
                n_reveal_this_round = 1  # always make progress while anything remains masked

            token_tensor = torch.tensor([tokens], device=device)
            mask_tensor = torch.tensor(
                [[1 if t == self.mask_token_id else 0 for t in tokens]], device=device
            )
            with torch.no_grad():
                logits, critic_scores = self.module(
                    token_tensor, mask_tensor, prompt_embedding=prompt_embedding
                )
                if guidance_scale > 0.0:
                    t_step = guidance_scale * (step / num_rounds)
                    uncond_logits, _ = self.module(
                        token_tensor, mask_tensor, prompt_embedding=uncond_embedding
                    )
                    logits = (1.0 + t_step) * logits - t_step * uncond_logits
            probs = torch.softmax(logits[0], dim=-1)
            predicted = probs.argmax(dim=-1)
            gen_confidence = probs.max(dim=-1).values
            # Token-Critic gating: don't trust the generator's own softmax
            # alone — combine with the critic head's independent score.
            # critic_scores stays from the CONDITIONAL forward pass only.
            combined_confidence = (gen_confidence * critic_scores[0]).tolist()

            candidates = [i for i in visit_order if tokens[i] == self.mask_token_id]
            candidates.sort(key=lambda i: combined_confidence[i], reverse=True)
            reveal_now = candidates[:n_reveal_this_round]

            for i in reveal_now:
                tokens[i] = int(predicted[i].item())
                confidence[i] = combined_confidence[i]

        return {"tokens": tokens, "confidence_map": confidence}
