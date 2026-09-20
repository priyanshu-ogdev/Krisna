# Phase 25 — Polish Cross-Stage Resolution: Researched, Flagged, Not Silently Fixed

Found while pairwise-diffing config values across stage boundaries within
the same tier (the highest-value remaining angle identified at the end
of Phase 24): `polish_default_lora_z_image.yaml` (the base LoRA
fine-tune) trains at `resolution: 1024`; `dpo_z_image_stage1_general.yaml`
(which continues training the *same* adapter via `--lora-adapter-path`)
trains at `resolution: 512`. No comment, citation, or acknowledgment
anywhere in the repo that these two stages of one continuous adapter
differ.

## What this phase did NOT do

Did not pick a "correct" value and silently change it. Neither number
had evidence attached either way before this phase — changing one
without evidence would just be introducing a different unverified claim
in place of the original one.

## What this phase did: research the actual risk, then document it precisely

**Architecture-level risk, checked against this project's own docs**:
`docs/architecture/RESEARCH_AND_CITATIONS.md` already documents
Z-Image-Turbo as "MMDiT lineage" — the FLUX/SD3 architecture family.
Researched (not assumed) what that family's positional encoding actually
does: FLUX/SD3-lineage MMDiT models use RoPE (Rotary Position
Embeddings) applied at every self-attention layer, not learned absolute
position embeddings. RoPE-based DiTs are specifically, repeatedly
documented in the literature as resolution-flexible by construction —
DyPE (arXiv:2510.20766) on training-free resolution extrapolation for
RoPE DiTs; FiT/FiTv2 explicitly described as "a flexible DiT trained on
multiple resolutions" as a normal, designed-in property, not a
workaround. This is a genuinely different, more inherently
resolution-tolerant mechanism than the Sketch tier's own **learned**
positional embeddings, which needed an explicit interpolation fix to
cross resolutions at all (Phase 24). So the architecture itself doesn't
make training two stages of the same adapter at different resolutions
dangerous by construction.

**Real-world precedent, not just architectural theory**: found a public
FLUX-LoRA training writeup (an MSc dissertation) that trained the
identical task at both 512 and 1024 resolution. The 1024 version was
explicitly the "primary deliverable"; the 512 version was reported to
behave "similarly at scale 0.5" but was explicitly the secondary
artifact, produced from infrastructure constraints, not the intended
main path. This is real, concrete evidence that (a) this exact situation
— one LoRA task trained at two resolutions — does happen in practice and
can work, and (b) it is nonetheless not treated as an equivalent,
interchangeable choice by practitioners who've actually done it; the
higher resolution is still preferred as canonical.

**Practitioner consensus, for balance**: a credible community discussion
(OneTrainer maintainer) states the opposite instinct as the safe
default — "train in the resolution the model you are training on
expects" — treating any deviation as a choice requiring its own
justification, not a free substitution.

## Net assessment

Very likely **safe** architecturally (RoPE-based MMDiT, not the fragile
learned-embedding case), but still **not confirmed as deliberate**.
Plausible, undocumented legitimate reasons this could be intentional:
DPO's per-step cost is inherently higher than a plain fine-tune's (both
a policy and a reference forward pass every step — see
`docs/review/18_training_memory_audit.md`), which could justify running
the preference-comparison phase at a cheaper resolution specifically.
But nothing in this repo states that reasoning anywhere. Both yaml files
now cross-reference each other with the full research above, so the
next person to touch either config sees the flag rather than a silent
mismatch — and can either document the real reason 512 was chosen, or
align it to 1024 if the difference was accidental drift.

## What this confirms about the review process at this point

This is the class of bug that only surfaces from a pairwise diff across
files in the same tier, not from reading any single file in isolation —
consistent with the prediction at the end of Phase 24. Worth continuing
that specific angle (cross-file parameter agreements within a tier, not
new subsystems) if further passes are wanted, rather than assuming
single-file review has reached diminishing returns project-wide.
