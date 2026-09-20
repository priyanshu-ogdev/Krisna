"""Exports the preference-pair store to standard DPO-ready JSONL files:
one JSON object per line, {"prompt": ..., "chosen": ..., "rejected": ...}.
"""

from __future__ import annotations

import json
from pathlib import Path

from krisna_training.preference.preference_store import PreferenceStore


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
