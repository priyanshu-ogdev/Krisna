"""LoRA target-module selection.

Standard Qwen/Llama-family attention+MLP projections are named
q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj almost universally
in HF implementations, and those names are used as the DEFAULT target
list below. But Qwen3.5's hybrid Gated DeltaNet + Gated Attention
architecture is new enough that I have not personally verified its exact
linear-layer names for the DeltaNet/SSM-style mixer layers — other
Gated-DeltaNet-family HF implementations (e.g. Qwen3-Next) have used
different naming (in_proj_qkvz, in_proj_ba, out_proj) for that mixer, and
I'm not going to bake in a guess I can't confirm.

So `discover_target_modules()` is the primary, robust path: introspect the
ACTUAL loaded model for nn.Linear submodules whose name matches a
suffix pattern, rather than trust a hard-coded list that might silently
match zero modules (or the wrong ones) against a real Qwen3.5 checkpoint.
Pass `include_patterns` to narrow it, or inspect
`list_linear_module_names(model)` yourself before training to see what's
actually there.
"""

from __future__ import annotations

DEFAULT_PATTERNS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def list_linear_module_names(model) -> list[str]:
    """Every nn.Linear submodule's dotted name in the model — inspect this
    directly if discover_target_modules()'s defaults don't look right for
    whatever Qwen3.5 checkpoint/transformers version you're actually on."""
    import torch.nn as nn

    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def discover_target_modules(model, include_patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> list[str]:
    """Returns the short (last-component) module names matching any of
    include_patterns, deduplicated — the format peft's LoraConfig.
    target_modules expects. Raises if nothing matched, rather than
    silently training a LoRA adapter with zero target modules (which peft
    will accept without complaint and produce an adapter that changes
    nothing)."""
    all_names = list_linear_module_names(model)
    matched = sorted({name.split(".")[-1] for name in all_names if any(p in name for p in include_patterns)})

    if not matched:
        raise ValueError(
            f"No Linear modules matched patterns {include_patterns} in this model. "
            f"Call list_linear_module_names(model) to see the actual module names "
            "and pass a corrected include_patterns — Qwen3.5's Gated DeltaNet mixer "
            "layer names have not been verified against these defaults."
        )
    return matched
