from __future__ import annotations

import pytest

from krisna_training.dpo.preference_store import PreferenceStore
from krisna_training.dpo.uicrit_importer import import_records


@pytest.fixture
def pref_store(tmp_path) -> PreferenceStore:
    s = PreferenceStore(db_path=tmp_path / "prefs.db")
    yield s
    s.close()


def test_import_groups_and_ranks(pref_store: PreferenceStore):
    records = [
        {"image_path": "blob://a.png", "rating": 4.5, "task_id": "t1", "prompt": "signup form"},
        {"image_path": "blob://b.png", "rating": 2.0, "task_id": "t1", "prompt": "signup form"},
        {"image_path": "blob://c.png", "rating": 3.0, "task_id": "t1", "prompt": "signup form"},
    ]
    written = import_records(pref_store, records)
    assert written == 1
    pairs = pref_store.list(source="uicrit_seed")
    assert len(pairs) == 1
    assert pairs[0].chosen_ref == "blob://a.png"
    assert pairs[0].rejected_ref == "blob://b.png"


def test_import_skips_ungrouped_singletons(pref_store: PreferenceStore):
    records = [{"image_path": "blob://a.png", "rating": 4.0, "task_id": "t1"}]
    written = import_records(pref_store, records)
    assert written == 0


def test_import_skips_small_score_gaps(pref_store: PreferenceStore):
    records = [
        {"image_path": "blob://a.png", "rating": 4.0, "task_id": "t1"},
        {"image_path": "blob://b.png", "rating": 3.9, "task_id": "t1"},
    ]
    written = import_records(pref_store, records, min_score_gap=0.5)
    assert written == 0


def test_import_handles_multiple_groups(pref_store: PreferenceStore):
    records = [
        {"image_path": "blob://a.png", "rating": 5.0, "task_id": "t1"},
        {"image_path": "blob://b.png", "rating": 1.0, "task_id": "t1"},
        {"image_path": "blob://c.png", "rating": 4.0, "task_id": "t2"},
        {"image_path": "blob://d.png", "rating": 1.0, "task_id": "t2"},
    ]
    written = import_records(pref_store, records)
    assert written == 2


def test_import_skips_records_missing_fields(pref_store: PreferenceStore, caplog):
    records = [
        {"image_path": "blob://a.png", "task_id": "t1"},  # missing rating
        {"rating": 3.0, "task_id": "t1"},  # missing image_path
    ]
    written = import_records(pref_store, records)
    assert written == 0


def test_import_falls_back_to_ungrouped_by_image_when_no_group_field(pref_store: PreferenceStore):
    records = [
        {"image_path": "blob://a.png", "rating": 4.0},
        {"image_path": "blob://b.png", "rating": 1.0},
    ]
    # No task_id field -> group_field falls back to image_path per record,
    # so each becomes its own singleton group -> nothing to pair.
    written = import_records(pref_store, records)
    assert written == 0
