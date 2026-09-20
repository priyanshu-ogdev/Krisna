"""DEPRECATED (final, no-RLHF-loop PRD revision): the Planner ships FROZEN.
This package's BF16 LoRA fine-tuning pipeline is NOT part of the active
training path — nothing in this project's current pipeline calls it, and
inference/planner_backend.py no longer has a lora_adapter_path parameter
at all (removed; see that module's docstring). Design-domain grounding
now comes from retrieval over data-forge's real UICrit corpus instead —
see inference/planner_rag.py.

Kept in the repository (not deleted) as a reference implementation, in
case a future PRD revision un-freezes the Planner — the LoRA-target-module
discovery logic (lora_config.py) and training-collation logic in this
package are still correct code, just not currently wired into anything
that runs. If reviving this path: (1) re-add lora_adapter_path to
PlannerBackend.__init__ and its load() method, (2) re-wire
KRISNA_PLANNER_LORA_PATH in inference/factory.py, (3) confirm this is a
deliberate architectural decision, not an accidental revert.

--- Original (v10 PRD) docstring, preserved for context ---
Planner training — BF16 LoRA fine-tuning for Qwen3.5-9B (§6.1: "trains
in BF16 LoRA specifically because Qwen3.5's hybrid Gated DeltaNet + Gated
Attention mechanism has a documented QLoRA degradation problem"). This
package NEVER quantizes for training — no bnb/NF4 config anywhere in here,
by design, matching that finding.

Produces a peft adapter directory (`model.save_pretrained(output_dir)`)
that `inference/planner_backend.py`'s existing `lora_adapter_path` /
`KRISNA_PLANNER_LORA_PATH` already knows how to load — that loading path
was built in the inference layer before this training code existed, and
this is what actually produces a real adapter for it to load.

Same lazy-import discipline as every other package here: torch/transformers
/peft only imported inside functions, never at module scope.
"""

import warnings

warnings.warn(
    "krisna_training.planner is deprecated under the final "
    "no-RLHF-loop PRD revision — the Planner ships frozen and is never "
    "fine-tuned by this project. See this package's __init__.py docstring.",
    DeprecationWarning,
    stacklevel=2,
)
