# models/ — Trained Checkpoint Artifacts

Not code — this is where trained checkpoints land, consumed by both
`inference/runtime/` (for manual testing) and by setting the corresponding
`KRISNA_*` environment variable when running the real service
(`scripts/inference/run_service.sh` / `.ps1`), per [docs/PRD.md](file:///d:/Krisna/docs/PRD.md) §6–§7.

```
models/
├── sketch_tier/               <- KRISNA_SKETCH_CHECKPOINT
│   └── checkpoint_final/      <- state_dict() + config, per
│                                  krisna_training.sketch.checkpoint_io
├── polish_default_lora/       <- KRISNA_POLISH_DEFAULT_LORA_PATH
│   └── <output_dir from configs/polish_default_lora_z_image.yaml>/
├── planner_rag_index/         <- KRISNA_PLANNER_RAG_CORPUS_DIR
│   └── model_data/planner_rag_corpus/uicrit_critiques.jsonl
│       (this is data-forge's model_data/ directory itself, or a copy of
│       just the planner_rag_corpus/ subtree — see docs/architecture/
│       SYNC_DESIGN.md. Not a "trained" artifact — retrieval corpus, not
│       a checkpoint — kept here anyway since it's the one non-frozen
│       input the Planner backend needs at load time.)
└── dpo_checkpoints/            <- KRISNA_POLISH_DEFAULT_LORA_PATH (deploy
                                    a DPO checkpoint the same way as the
                                    base LoRA — see docs/review/11_scripts_review.md
                                    for why it's the same env var, not a
                                    separate one)
                                    Stage 1 general + Stage 2 domain, per
                                    data-forge's dpo_alignment/{general,
                                    domain}/ split. A real, tested trainer
                                    exists (training/src/krisna_training/
                                    polish/train_dpo.py, launched via
                                    scripts/training/train_polish_dpo.sh)
                                    — this directory just ships empty
                                    until someone actually runs it, same
                                    as every other subdirectory here.
```

Each subdirectory's own checkpoint format is whatever the training code
that produces it actually writes — see the corresponding section of
`training/README.md` for the exact shape (e.g. sketch tier saves
`state_dict()` + config, not a pickled model object; polish-default LoRA
is whatever `diffusers`' official training script writes via
`accelerate launch`).

**This directory ships empty in the repo** (checkpoints are large binary
artifacts, not source) — populate it by actually running the training
scripts in `training/README.md`, or point the `KRISNA_*` env vars at
wherever your checkpoints really live instead of moving them here. The
directory structure above is a convention `inference/runtime/` defaults
to, not a hard requirement.
