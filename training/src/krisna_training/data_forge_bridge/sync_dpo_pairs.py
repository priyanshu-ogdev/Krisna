"""Imports data-forge's raw preference pairs (`preference_pairs/{source}/
*.json` + the image_a/image_b files next to each) into this project's
`dpo.preference_store.PreferenceStore` as real `PreferencePair` rows.

Reads the PRE-latent-encoding stage (data-forge's `s01_6_preference_pairs`
output), not `s08_5_dpo_encoding.py`'s `.safetensors` latents. This is
still correct even now that `polish/train_dpo.py` exists as the real DPO
trainer: `train_dpo.py` resolves `chosen_ref`/`rejected_ref` through the
shared `BlobStore` too, i.e. it also consumes images and encodes them
itself at train time, not `s08_5_dpo_encoding.py`'s precomputed latents
— so importing images here (rather than waiting on a latent-consuming
path) is still the right call, just for a different reason than
originally written: not because no trainer exists, but because the real
trainer doesn't consume that artifact either. (`s08_5_dpo_encoding.py`'s
output remains unconsumed anywhere in this project — see
`docs/review/02_polish_tier.md` for the full design-sync review of this.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from krisna_training.dpo.preference_store import PreferencePair, PreferenceStore

log = logging.getLogger("krisna_training.data_forge_bridge.sync_dpo_pairs")

# data-forge's preference_pairs/ subfolder names -> this project's
# preference_store.VALID_SOURCES entries. A 1:1 mapping today, kept
# explicit (not just "use the folder name directly") so a data-forge-side
# rename doesn't silently produce an invalid source value here — it'll
# raise KeyError instead, loud and immediate.
_SOURCE_KEY_MAP = {
    "pickapic_v2": "pickapic_v2",
    "hpdv2": "hpdv2",
    "designsense_10k": "designsense_10k",
    "designpref": "designpref",
}


def sync(
    data_forge_data_root: str | Path,
    preference_store: PreferenceStore,
    blob_store,
    sources: list[str] | None = None,
) -> dict[str, int]:
    """data_forge_data_root: data-forge's DATA_ROOT (parent of
    `preference_pairs/`, NOT `model_data/` — this reads the raw stage).
    blob_store: an inference.common.BlobStore (or
    blob_store_singleton.get_blob_store()) to copy image_a/image_b into.
    sources: which data-forge subfolders to import; defaults to all four
    mapped keys.

    Returns {source_key: pairs_imported} plus a "skipped_not_deduped" and
    "skipped_missing_images" count.
    """
    pref_root = Path(data_forge_data_root) / "preference_pairs"
    if not pref_root.exists():
        raise FileNotFoundError(
            f"{pref_root} not found — pass data-forge's DATA_ROOT, not model_data_root "
            "(this reads the raw pre-latent-encoding stage)."
        )

    sources = sources or list(_SOURCE_KEY_MAP)
    counts: dict[str, int] = {}
    skipped_not_deduped = 0
    skipped_missing_images = 0

    for source_dir_name in sources:
        if source_dir_name not in _SOURCE_KEY_MAP:
            raise KeyError(
                f"Unknown source '{source_dir_name}' — not in _SOURCE_KEY_MAP. "
                "If data-forge added a new preference-pair source, add it here "
                "AND to preference_store.VALID_SOURCES deliberately, not silently."
            )
        mapped_source = _SOURCE_KEY_MAP[source_dir_name]
        source_dir = pref_root / source_dir_name
        if not source_dir.exists():
            counts[mapped_source] = 0
            continue

        imported = 0
        for meta_path in sorted(source_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("sync_dpo_bad_metadata", extra={"path": str(meta_path), "error": str(e)})
                continue

            if meta.get("dedup_status") != "unique":
                # Same rule s08_5_dpo_encoding.py applies — never processed
                # by data-forge's own s01_6 dedup/safety pass, so it's not
                # safe to treat as a clean pair.
                skipped_not_deduped += 1
                continue

            image_a_path = source_dir / meta["image_a"]
            image_b_path = source_dir / meta["image_b"]
            if not image_a_path.exists() or not image_b_path.exists():
                skipped_missing_images += 1
                continue

            from PIL import Image

            ref_a = blob_store.save_image(Image.open(image_a_path), prefix=f"{mapped_source}_a")
            ref_b = blob_store.save_image(Image.open(image_b_path), prefix=f"{mapped_source}_b")

            preferred = meta.get("preferred")
            if preferred == "a":
                chosen_ref, rejected_ref = ref_a, ref_b
            elif preferred == "b":
                chosen_ref, rejected_ref = ref_b, ref_a
            else:
                log.warning(
                    "sync_dpo_missing_preferred_label",
                    extra={"pair_id": meta.get("pair_id"), "preferred": preferred},
                )
                continue

            try:
                pair = PreferencePair(
                    prompt=meta.get("prompt", ""),
                    chosen_ref=chosen_ref,
                    rejected_ref=rejected_ref,
                    source=mapped_source,
                )
            except ValueError as e:
                log.warning("sync_dpo_pair_construction_failed", extra={"pair_id": meta.get("pair_id"), "error": str(e)})
                continue

            preference_store.add(pair)
            imported += 1

        counts[mapped_source] = imported

    counts["skipped_not_deduped"] = skipped_not_deduped
    counts["skipped_missing_images"] = skipped_missing_images
    log.info("sync_dpo_pairs_complete", extra=counts)
    return counts
