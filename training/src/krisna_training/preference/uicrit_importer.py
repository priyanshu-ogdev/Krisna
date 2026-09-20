"""UICrit seed-data importer.

Parses parsed human ratings over UI designs and groups them into preference pairs.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from krisna_training.preference.pair_builder import rank_candidates
from krisna_training.preference.preference_store import PreferencePair, PreferenceStore

log = logging.getLogger("krisna_training.preference.uicrit_importer")


def import_records(
    store: PreferenceStore,
    records: list[dict],
    image_field: str = "image_path",
    rating_field: str = "rating",
    group_field: str = "task_id",
    prompt_field: str | None = "prompt",
    min_score_gap: float = 0.5,
) -> int:
    """records: list of plain dicts as parsed from UICrit's dataset file.
    Groups entries by `group_field` and pairs the highest- and lowest-rated records.
    Returns the number of pairs written.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if image_field not in r or rating_field not in r:
            log.warning(
                "uicrit_record_missing_fields",
                extra={"expected": [image_field, rating_field], "got": list(r.keys())},
            )
            continue
        key = r.get(group_field, r[image_field])
        groups[key].append(r)

    written = 0
    for group_key, items in groups.items():
        if len(items) < 2:
            continue
        candidates = [{"image_ref": it[image_field], "score": float(it[rating_field])} for it in items]
        ranked = rank_candidates(candidates)
        best, worst = ranked[0], ranked[-1]
        if best["image_ref"] == worst["image_ref"]:
            continue
        if (best["score"] - worst["score"]) < min_score_gap:
            continue

        prompt = "a mobile UI screen"
        if prompt_field and prompt_field in items[0] and items[0][prompt_field]:
            prompt = str(items[0][prompt_field])

        pair = PreferencePair(
            prompt=prompt,
            chosen_ref=best["image_ref"],
            rejected_ref=worst["image_ref"],
            chosen_score=best["score"],
            rejected_score=worst["score"],
            source="uicrit_seed",
            session_id=str(group_key),
        )
        store.add(pair)
        written += 1

    return written
