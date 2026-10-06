"""UICrit seed-data importer — §6: "seeded by UICrit + synthetic + Gemma-
derived pairs."

UICrit (registered in data-forge's datasets.yaml as
google-research-datasets/uicrit, CC BY-ND) provides human ratings over UI
designs. This module does NOT hard-code UICrit's exact column/field names
— I have not personally inspected the dataset's actual CSV/JSON schema, and
guessing wrong field names here would silently produce zero pairs rather
than failing loudly. Instead, `import_records()` takes an already-parsed
list of plain dicts and a small set of field-name overrides, so the person
wiring this up points it at UICrit's real columns once, explicitly, rather
than this module assuming names it hasn't verified. (This is the same
posture as the sketch tier's "no checkpoint" note — better to be honest
about what's unverified than to fabricate a schema.)

Records are grouped by `group_key` (the same UI/task being rated multiple
times, e.g. by different annotators or across design variants) and the
group's highest- and lowest-rated entries become a preference pair —
same ranking logic as pair_builder.rank_candidates, reused here.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from krisna_training.dpo.pair_builder import rank_candidates
from krisna_training.dpo.preference_store import PreferencePair, PreferenceStore

log = logging.getLogger("krisna_training.dpo.uicrit_importer")


def import_records(
    store: PreferenceStore,
    records: list[dict],
    image_field: str = "image_path",
    rating_field: str = "rating",
    group_field: str = "task_id",
    prompt_field: str | None = "prompt",
    min_score_gap: float = 0.5,
) -> int:
    """records: list of plain dicts as parsed from UICrit's own file
    (CSV/JSON — parsing that file is the caller's job; this only handles
    grouping + pairing). Field name parameters default to plausible
    UICrit-style names but are NOT verified against the actual dataset —
    override them once you've checked the real schema.

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
        key = r.get(group_field, r[image_field])  # fall back to ungrouped (1 record = 1 group)
        groups[key].append(r)

    written = 0
    for key, group_records in groups.items():
        if len(group_records) < 2:
            continue

        candidates = [
            {"image_ref": r[image_field], "score": float(r[rating_field])} for r in group_records
        ]
        ranked = rank_candidates(candidates)
        best, worst = ranked[0], ranked[-1]
        if best["image_ref"] == worst["image_ref"]:
            continue
        if (best["score"] - worst["score"]) < min_score_gap:
            continue

        prompt = group_records[0].get(prompt_field, "") if prompt_field else ""
        pair = PreferencePair(
            prompt=prompt or f"UICrit group {key}",
            chosen_ref=best["image_ref"],
            rejected_ref=worst["image_ref"],
            chosen_score=best["score"],
            rejected_score=worst["score"],
            source="uicrit_seed",
        )
        store.add(pair)
        written += 1

    log.info("uicrit_import_complete", extra={"groups": len(groups), "pairs_written": written})
    return written
