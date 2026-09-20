# Phase 24 — Data Pipeline & Training Final Check: Tensor-Shape/Dimension Consistency

Scoped specifically to this pass's ask: tensor and dimension mismatches
across the data pipeline → training boundary and within the training
code itself. Checked the highest-risk spots — the places where a shape
bug would either silently degrade output quality or crash — rather than
re-deriving the whole pipeline from scratch.

## Checked and confirmed correct

- **Sketch tier grid dimensions across stages**: Stage 1 (16×16,
  matching 256px at VQGAN's f16 downsample) → Stage 2 (32×32, matching
  512px). `vocab_size=16384` consistent across both, matching
  `boris/vqgan_f16_16384`'s real codebook.
- **Progressive-resolution positional-embedding interpolation**
  (`pos_embed.py`): reshapes `[H*W, dim]` → `[1, H, W, dim]` → permutes
  to NCHW for `F.interpolate(mode="bicubic")` → permutes back and
  reshapes to `[new_H*new_W, dim]`. Correct dimension ordering, no
  off-by-one. Confirmed `pos_embed` is added to token embeddings
  *before* the prefix-conditioning `torch.cat` in `model.py`'s forward
  pass — meaning the positional embedding tensor covers only the H×W
  grid, not a prefix-token slot, so treating the whole tensor as pure
  grid positions during interpolation is correct, not a semantic bug
  hiding behind a shape that happens to work.
- **A runtime shape assertion already exists** (`model.py`:
  `assert N == self.cfg.seq_len`) catching any token-count mismatch at
  forward-pass time, independent of the checks above.
- **CFG conditioning-dropout collate function**: `make_collate_fn`'s
  zero-embedding fallback (`torch.zeros(len(batch), prompt_dim)`) uses
  the same `cfg.prompt_dim` passed to the model, no separate hardcoded
  dimension to drift from it.
- **DPO latent scaling**: `train_dpo.py` reads `vae.config.scaling_factor`
  from the actually-loaded VAE at runtime rather than hardcoding a
  value (e.g. Stable Diffusion's typical 0.18215) that would silently
  be wrong for Z-Image-Turbo's own VAE.

## Found and fixed: a stale placeholder dimension, defense-in-depth but wrong

`sketch_backend.py`'s CFG-conditioning code (added in Phase 17) already
has a genuinely good defensive pattern: it compares the live CLIP
embedder's output dimension against the checkpoint's saved `prompt_dim`
and falls back to an unconditional zero embedding (with
`guidance_scale=0.0`) rather than crashing on a shape mismatch —

```python
prompt_dim = getattr(self._model.module.cfg, "prompt_dim", 4096)
...
if prompt_embedding.shape[-1] != prompt_dim:
    ...
    prompt_embedding = torch.zeros(1, prompt_dim, device=device)
```

The fallback constant, `4096`, was wrong. `openai/clip-vit-large-patch14`
(the model `get_clip_embedder()` actually loads) has a real projection
dimension of **768**, not 4096. Both real training configs
(`sketch_train_stage1_256.yaml`, `sketch_train_stage2_512.yaml`)
correctly set `prompt_dim: 768` with an explicit comment confirming
it's deliberate — but `model.py`'s own dataclass default was still
`4096`, with a comment claiming it "matches the placeholder in
inference/sketch_backend.py," which was circular: both were leftover
values from before Phase 17's real CLIP integration existed at all,
and neither was ever updated once the real value (768) was established
everywhere that actually matters.

**Why this wasn't a live bug, but was a real landmine**: every checkpoint
produced by the current training configs carries `prompt_dim=768` in
its own saved config, so `getattr(self._model.module.cfg, "prompt_dim",
4096)` retrieves the real, correct 768 in practice — the `4096` fallback
never actually executes for a real checkpoint. It would only fire for a
checkpoint predating this field's existence, or a hand-constructed
config missing it. In that specific edge case, the fallback would
compare CLIP's real 768-dim output against the wrong 4096, the mismatch
guard would trigger, and generation would silently degrade to
unconditional (defeating Phase 17's fix) — for the wrong reason
(a stale default) rather than a genuine dimension mismatch.

**Fix**: both defaults corrected to `768`, with the reasoning above
recorded in-line in each file so a future reader — or session — has the
context rather than a silent number change.

## Not independently re-verified this pass (already covered by earlier phases, not re-derived)

- Z-Image-Turbo's own internal latent channel count and DiT
  hidden-dimension consistency — this project treats these as opaque to
  `diffusers`' own pipeline code (loaded via `from_pretrained`, never
  hardcoded dimensions on this project's side), which is the correct
  posture rather than something to second-guess without the actual
  model loaded.
- Critic (Gemma 4) tensor shapes — out of scope for training-side
  tensor-mismatch review, since Critic ships frozen and is never
  trained by this project (Phase 4/11).

## Verification

Full suite re-run after both fixes: **356 passed, 10 skipped**, same as
every prior phase in this session — the fix is defensive-code-only and
touches no path exercised by MockBackend-based tests, consistent with
why no test needed updating. Full repo syntax sweep also re-run,
clean.

## Findings summary (this phase)

| Severity | Finding | Status |
|---|---|---|
| **Real, fixed (Low severity — dormant except on legacy/malformed checkpoints)** | `sketch_backend.py`'s CFG shape-mismatch-guard fallback and `model.py`'s `prompt_dim` dataclass default both used a stale placeholder (4096) instead of CLIP ViT-L/14's real 768-dim output, which both real training configs already correctly use | Fixed — both defaults corrected to 768, reasoning recorded in-line |
| Info | Positional-embedding interpolation, grid-dimension consistency across training stages, CFG collate-function shapes, and DPO VAE scaling-factor handling | All confirmed correct on direct inspection — no bugs found |
