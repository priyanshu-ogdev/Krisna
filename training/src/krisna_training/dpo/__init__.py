"""DPO preference-pair pipeline — PRD §6: "Diffusion-DPO on ranked candidate
pairs ... small-batch, seeded by UICrit + synthetic + Gemma-derived pairs."

This builds and stores (chosen, rejected) image-ref pairs from three
sources — verifier-stack rankings, Gemma critique scores, and imported
UICrit human ratings — and exports them in a standard DPO training format.
It does NOT implement the actual DPO training loop; that's a separate
training-pipeline build, out of scope here.
"""
