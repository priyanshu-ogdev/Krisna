"""DEPRECATED (final, no-RLHF-loop PRD revision): the Critic (Gemma 4)
ships FROZEN as an on-demand product feature. This package's QLoRA
training pipeline is NOT part of the active training path — nothing in
this project's current pipeline calls it, and inference/critic_worker.py
/critic_backend.py no longer have a lora_adapter_path parameter at all
(removed; see critic_worker.py's module docstring). There is no AI-judge
critique-labeled training data generated anywhere in this pipeline.

Kept in the repository (not deleted) as a reference implementation, in
case a future PRD revision un-freezes the Critic — masking.py and
dataset.py's logic is still correct code, just not currently wired into
anything that runs. If reviving this path: (1) re-add lora_adapter_path
to CriticBackend.__init__/critic_worker.py's _load(), (2) re-wire
KRISNA_CRITIC_LORA_PATH in inference/factory.py, (3) revert the
"gemma4_31b_frozen" critique_source naming back to something reflecting a
trained checkpoint, (4) confirm this is a deliberate architectural
decision, not an accidental revert.

--- Original (v10 PRD) docstring, preserved for context ---
Critic (Gemma 4 31B Dense) QLoRA training via Unsloth.

Same isolation constraint as critic_worker.py / critic_backend.py (see
inference/critic_worker.py's module docstring for the full explanation):
this MUST run inside ./venv-critic, not the main venv — unsloth_zoo caps
transformers<=5.5.0, which conflicts with the git-main transformers build
the Planner tier needs. `masking.py` and `dataset.py` in this package have
ZERO external dependencies beyond the standard library (masking.py) or
whatever tokenizer/processor object is passed in (dataset.py takes it as a
parameter, never imports transformers itself) — they're written this way
specifically so they're testable in the main pytest venv (no
unsloth/transformers==5.5.0 needed) while still being the exact same code
`train_critic_qlora.py` uses when actually run under venv-critic.

`train_critic_qlora.py` itself DOES need unsloth/transformers==5.5.0 (it
imports FastModel), so it can only run via:
    ./venv-critic/bin/python -m krisna_training.critic.train_critic_qlora --config ...
which requires krisna_training (formerly krisna_orchestrator) itself importable from venv-critic —
`scripts/setup_env_critic.sh` handles this with `pip install -e . --no-deps`
(the package's own code, without pulling in fastapi/pydantic/etc., which
venv-critic never needs).
"""

import warnings

warnings.warn(
    "krisna_training.critic is deprecated under the final "
    "no-RLHF-loop PRD revision — the Critic ships frozen and is never "
    "trained by this project. See this package's __init__.py docstring.",
    DeprecationWarning,
    stacklevel=2,
)
