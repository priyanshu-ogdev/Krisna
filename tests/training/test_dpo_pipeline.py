from __future__ import annotations

import pytest

from krisna_training.dpo.pair_builder import (
    aggregate_verifier_score,
    build_pair_from_candidates,
    rank_candidates,
)
from krisna_training.dpo.preference_store import PreferencePair, PreferenceStore


@pytest.fixture
def pref_store(tmp_path) -> PreferenceStore:
    s = PreferenceStore(db_path=tmp_path / "prefs.db")
    yield s
    s.close()


def test_aggregate_verifier_score_ignores_none():
    scores = {"a": 0.8, "b": None, "c": 0.6}
    assert aggregate_verifier_score(scores) == pytest.approx(0.7)


def test_aggregate_verifier_score_all_none_returns_none():
    assert aggregate_verifier_score({"a": None, "b": None}) is None


def test_rank_candidates_highest_first():
    candidates = [
        {"image_ref": "a", "score": 0.3},
        {"image_ref": "b", "score": 0.9},
        {"image_ref": "c", "score": 0.6},
    ]
    ranked = rank_candidates(candidates)
    assert [c["image_ref"] for c in ranked] == ["b", "c", "a"]


def test_rank_candidates_none_scores_sort_last():
    candidates = [{"image_ref": "a", "score": None}, {"image_ref": "b", "score": 0.5}]
    ranked = rank_candidates(candidates)
    assert ranked[0]["image_ref"] == "b"
    assert ranked[1]["image_ref"] == "a"


def test_preference_pair_rejects_invalid_source():
    with pytest.raises(ValueError):
        PreferencePair(prompt="x", chosen_ref="a", rejected_ref="b", source="not_a_real_source")


def test_preference_pair_rejects_identical_refs():
    with pytest.raises(ValueError):
        PreferencePair(prompt="x", chosen_ref="a", rejected_ref="a", source="verifier_stack")


def test_store_add_and_list(pref_store: PreferenceStore):
    pair = PreferencePair(
        prompt="minimalist login screen", chosen_ref="blob://a.png", rejected_ref="blob://b.png",
        chosen_score=0.9, rejected_score=0.5, source="verifier_stack",
    )
    pref_store.add(pair)
    all_pairs = pref_store.list()
    assert len(all_pairs) == 1
    assert all_pairs[0].chosen_ref == "blob://a.png"


def test_store_list_filters_by_source(pref_store: PreferenceStore):
    pref_store.add(PreferencePair(prompt="x", chosen_ref="a", rejected_ref="b", source="verifier_stack"))
    pref_store.add(PreferencePair(prompt="y", chosen_ref="c", rejected_ref="d", source="uicrit_seed"))
    assert pref_store.count(source="verifier_stack") == 1
    assert pref_store.count(source="uicrit_seed") == 1
    assert pref_store.count() == 2


def test_build_pair_from_candidates_picks_best_and_worst(pref_store: PreferenceStore):
    candidates = [
        {"image_ref": "blob://a.png", "score": 0.9},
        {"image_ref": "blob://b.png", "score": 0.3},
        {"image_ref": "blob://c.png", "score": 0.6},
    ]
    pair = build_pair_from_candidates(pref_store, prompt="a poster", candidates=candidates, source="verifier_stack")
    assert pair is not None
    assert pair.chosen_ref == "blob://a.png"
    assert pair.rejected_ref == "blob://b.png"
    assert pref_store.count() == 1


def test_build_pair_from_candidates_skips_near_ties(pref_store: PreferenceStore):
    candidates = [
        {"image_ref": "blob://a.png", "score": 0.71},
        {"image_ref": "blob://b.png", "score": 0.70},
    ]
    pair = build_pair_from_candidates(
        pref_store, prompt="x", candidates=candidates, source="verifier_stack", min_score_gap=0.1
    )
    assert pair is None
    assert pref_store.count() == 0


def test_build_pair_from_candidates_skips_when_fewer_than_two_scored(pref_store: PreferenceStore):
    candidates = [{"image_ref": "blob://a.png", "score": 0.9}, {"image_ref": "blob://b.png", "score": None}]
    pair = build_pair_from_candidates(pref_store, prompt="x", candidates=candidates, source="verifier_stack")
    assert pair is None


def test_export_jsonl(pref_store: PreferenceStore, tmp_path):
    from krisna_training.dpo.dpo_dataset_export import export_jsonl

    pref_store.add(PreferencePair(prompt="x", chosen_ref="a", rejected_ref="b", source="verifier_stack"))
    pref_store.add(PreferencePair(prompt="y", chosen_ref="c", rejected_ref="d", source="uicrit_seed"))

    out_path = tmp_path / "dpo.jsonl"
    count = export_jsonl(pref_store, out_path)
    assert count == 2

    import json

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert set(first.keys()) == {"prompt", "chosen", "rejected", "source", "chosen_score", "rejected_score"}


def test_export_jsonl_filters_by_source(pref_store: PreferenceStore, tmp_path):
    from krisna_training.dpo.dpo_dataset_export import export_jsonl

    pref_store.add(PreferencePair(prompt="x", chosen_ref="a", rejected_ref="b", source="verifier_stack"))
    pref_store.add(PreferencePair(prompt="y", chosen_ref="c", rejected_ref="d", source="uicrit_seed"))

    out_path = tmp_path / "dpo_uicrit_only.jsonl"
    count = export_jsonl(pref_store, out_path, source="uicrit_seed")
    assert count == 1
