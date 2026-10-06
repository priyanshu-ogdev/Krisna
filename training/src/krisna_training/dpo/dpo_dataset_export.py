"""Exports the preference-pair store to a DPO-ready JSONL file: one JSON
object per line, {"prompt": ..., "chosen": ..., "rejected": ...} — the
field names `trl`'s DPOTrainer and most diffusion-DPO training scripts
expect out of the box. `chosen`/`rejected` are the stored blob:// image
refs, not raw image data — resolving those to actual files at training
time is the training pipeline's job, not this export step's.
"""

from __future__ import annotations

import json
from pathlib import Path

from krisna_training.dpo.preference_store import PreferenceStore


def export_jsonl(
    store: PreferenceStore,
    output_path: str | Path,
    source: str | None = None,
    session_id: str | None = None,
) -> int:
    """Returns the number of pairs written."""
    pairs = store.list(source=source, session_id=session_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(
                json.dumps(
                    {
                        "prompt": pair.prompt,
                        "chosen": pair.chosen_ref,
                        "rejected": pair.rejected_ref,
                        "source": pair.source,
                        "chosen_score": pair.chosen_score,
                        "rejected_score": pair.rejected_score,
                    }
                )
                + "\n"
            )
    return len(pairs)
