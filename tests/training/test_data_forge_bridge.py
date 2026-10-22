from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from krivex_inference.backends.common import BlobStore
from krivex_training.data_forge_bridge.sync_dpo_pairs import sync as sync_dpo
from krivex_training.data_forge_bridge.sync_planner_rag import count_critiques, sync as sync_planner
from krivex_training.data_forge_bridge.sync_polish_default import sync as sync_polish
from krivex_training.dpo.preference_store import PreferenceStore


# ---------------------------------------------------------------------------
# Planner RAG Corpus Bridge Tests
# ---------------------------------------------------------------------------

def test_sync_planner_rag_copies_and_counts(tmp_path):
    model_data = tmp_path / "model_data"
    rag_source = model_data / "planner_rag_corpus"
    rag_source.mkdir(parents=True)

    jsonl_file = rag_source / "uicrit_critiques.jsonl"
    entries = [
        {"record_id": "r1", "critique_output": {"summary": "good"}},
        {"record_id": "r2", "critique_output": {"summary": "needs spacing"}},
    ]
    jsonl_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    out_dir = tmp_path / "data"
    synced = sync_planner(model_data, out_dir)

    assert synced.exists()
    assert synced.name == "uicrit_critiques.jsonl"
    assert count_critiques(synced) == 2


def test_sync_planner_rag_missing_source_raises(tmp_path):
    empty_model_data = tmp_path / "empty_model_data"
    empty_model_data.mkdir()
    with pytest.raises(FileNotFoundError, match="UICrit critiques file not found"):
        sync_planner(empty_model_data, tmp_path / "out")


# ---------------------------------------------------------------------------
# DPO Preference Pairs Bridge Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def pref_store(tmp_path) -> PreferenceStore:
    s = PreferenceStore(db_path=tmp_path / "prefs.db")
    yield s
    s.close()


@pytest.fixture
def blob_store(tmp_path) -> BlobStore:
    return BlobStore(root=tmp_path / "blobs")


def _write_pair(source_dir, pair_id, preferred="a", dedup_status="unique", prompt="a poster"):
    source_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="red").save(source_dir / f"{pair_id}_a.png")
    Image.new("RGB", (4, 4), color="blue").save(source_dir / f"{pair_id}_b.png")
    meta = {
        "pair_id": pair_id, "image_a": f"{pair_id}_a.png", "image_b": f"{pair_id}_b.png",
        "preferred": preferred, "dedup_status": dedup_status, "prompt": prompt,
        "safety_tier": "safe", "pii_scrubbed": True, "text_pii_scrubbed": True,
    }
    (source_dir / f"{pair_id}.json").write_text(json.dumps(meta))


def test_sync_imports_pairs_with_correct_chosen_rejected(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "pickapic_v2", "p1", preferred="a")

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["pickapic_v2"])
    assert counts["pickapic_v2"] == 1

    pairs = pref_store.list(source="pickapic_v2")
    assert len(pairs) == 1
    assert pairs[0].chosen_ref.startswith("blob://pickapic_v2_a_")
    assert pairs[0].rejected_ref.startswith("blob://pickapic_v2_b_")
    assert pairs[0].prompt == "a poster"


def test_sync_preferred_b_swaps_chosen_rejected(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "hpdv2", "p1", preferred="b")

    sync_dpo(data_root, pref_store, blob_store, sources=["hpdv2"])
    pairs = pref_store.list(source="hpdv2")
    assert pairs[0].chosen_ref.startswith("blob://hpdv2_b_")
    assert pairs[0].rejected_ref.startswith("blob://hpdv2_a_")


def test_sync_skips_non_deduped_pairs(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "designsense_10k", "p1", dedup_status="pending")

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["designsense_10k"])
    assert counts["designsense_10k"] == 0
    assert counts["skipped_not_deduped"] == 1
    assert pref_store.count() == 0


def test_sync_skips_pairs_missing_preferred_label(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "designpref"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(source_dir / "p1_a.png")
    Image.new("RGB", (4, 4)).save(source_dir / "p1_b.png")
    (source_dir / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "image_a": "p1_a.png", "image_b": "p1_b.png",
        "dedup_status": "unique",
        "safety_tier": "safe", "pii_scrubbed": True, "text_pii_scrubbed": True,
    }))

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["designpref"])
    assert counts["designpref"] == 0
    assert pref_store.count() == 0


def test_sync_skips_missing_image_files(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "pickapic_v2"
    source_dir.mkdir(parents=True)
    (source_dir / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "image_a": "missing_a.png", "image_b": "missing_b.png",
        "dedup_status": "unique", "preferred": "a",
        "safety_tier": "safe", "pii_scrubbed": True, "text_pii_scrubbed": True,
    }))

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["pickapic_v2"])
    assert counts["skipped_missing_images"] == 1
    assert pref_store.count() == 0


def test_sync_rejects_path_traversal(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "pickapic_v2"
    source_dir.mkdir(parents=True)
    (source_dir / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "image_a": "..", "image_b": "p1_b.png",
        "preferred": "a", "dedup_status": "unique",
        "safety_tier": "safe", "pii_scrubbed": True, "text_pii_scrubbed": True,
    }))

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["pickapic_v2"])
    assert counts["skipped_missing_images"] == 1
    assert pref_store.count() == 0


def test_sync_missing_source_dir_returns_zero_not_error(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    counts = sync_dpo(data_root, pref_store, blob_store, sources=["hpdv2"])
    assert counts["hpdv2"] == 0


def test_sync_missing_preference_pairs_root_raises(tmp_path, pref_store, blob_store):
    with pytest.raises(FileNotFoundError):
        sync_dpo(tmp_path / "nonexistent", pref_store, blob_store)


def test_sync_unknown_source_key_raises(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    with pytest.raises(KeyError):
        sync_dpo(data_root, pref_store, blob_store, sources=["totally_unknown_source"])


def test_sync_multiple_pairs_same_source(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "hpdv2"
    _write_pair(source_dir, "p1")
    _write_pair(source_dir, "p2")

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["hpdv2"])
    assert counts["hpdv2"] == 2
    assert pref_store.count(source="hpdv2") == 2


def test_sync_gamelabel_10k_pair(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "gamelabel_10k", "gl1", preferred="a")

    counts = sync_dpo(data_root, pref_store, blob_store, sources=["gamelabel_10k"])
    assert counts["gamelabel_10k"] == 1
    assert pref_store.count(source="gamelabel_10k") == 1
    pairs = pref_store.list(source="gamelabel_10k")
    assert pairs[0].source == "gamelabel_10k"


def test_sync_default_sources_covers_all_five(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    counts = sync_dpo(data_root, pref_store, blob_store)
    for key in ("pickapic_v2", "hpdv2", "gamelabel_10k", "designsense_10k", "designpref"):
        assert key in counts


# ---------------------------------------------------------------------------
# Polish Default Bridge Tests
# ---------------------------------------------------------------------------

def _make_data_forge_export(root, records):
    zimage_dir = root / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    images_dir.mkdir(parents=True)
    captions = []
    for record in records:
        record_id, image_filename, caption = record[0], record[1], record[2]
        source_caption = record[3] if len(record) > 3 else None
        Image.new("RGB", (4, 4), color="green").save(images_dir / image_filename)
        captions.append({
            "record_id": record_id, "caption": caption, "image_filename": image_filename,
            "source_caption": source_caption,
        })
    (zimage_dir / "captions.jsonl").write_text("\n".join(json.dumps(c) for c in captions))
    return root


def test_sync_bridges_record_keyed_captions_to_filename_keyed(tmp_path):
    model_data = _make_data_forge_export(
        tmp_path / "model_data",
        [
            ("uuid-aaa-111", "pd12m_0000042.png", "a minimalist login screen"),
            ("uuid-bbb-222", "cc12m_0000091.png", "a bold poster"),
        ],
    )
    out = sync_polish(model_data, tmp_path / "prepared")

    metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    by_name = {r["file_name"]: r["text"] for r in metadata_lines}
    assert by_name == {
        "pd12m_0000042.png": "a minimalist login screen",
        "cc12m_0000091.png": "a bold poster",
    }
    assert (out / "pd12m_0000042.png").exists()
    assert (out / "cc12m_0000091.png").exists()


def test_sync_missing_images_dir_raises_clear_error(tmp_path):
    model_data = tmp_path / "model_data"
    (model_data / "polish_zimage_turbo").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="s12_model_data_export"):
        sync_polish(model_data, tmp_path / "prepared")


def test_sync_refuses_missing_caption_allowlist(tmp_path):
    zimage_dir = tmp_path / "model_data" / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(images_dir / "webui_0000007.png")
    with pytest.raises(FileNotFoundError, match="caption allowlist"):
        sync_polish(tmp_path / "model_data", tmp_path / "prepared")


class TestRecordIdVsFilenameBugFix:
    def test_captions_match_via_image_filename_not_record_id_guess(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("f47ac10b-58cc-4372-a567-0e02b2c3d479", "rico_semantic_0009981.png", "a settings panel")],
        )
        out = sync_polish(model_data, tmp_path / "prepared")
        metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
        by_name = {r["file_name"]: r["text"] for r in metadata_lines}
        assert by_name["rico_semantic_0009981.png"] == "a settings panel"

    def test_legacy_captions_format_without_image_filename_is_rejected(self, tmp_path):
        zimage_dir = tmp_path / "model_data" / "polish_zimage_turbo"
        images_dir = zimage_dir / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(images_dir / "some_real_filename.png")
        (zimage_dir / "captions.jsonl").write_text(
            json.dumps({"record_id": "uuid-abc", "caption": "will not match"})
        )

        with pytest.raises(ValueError, match="current export format"):
            sync_polish(tmp_path / "model_data", tmp_path / "prepared")

    def test_orphaned_export_images_are_not_added_to_training(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("uuid-1", "current.png", "an allowed image")],
        )
        Image.new("RGB", (4, 4), color="red").save(
            model_data / "polish_zimage_turbo" / "images" / "orphan.png"
        )

        out = sync_polish(model_data, tmp_path / "prepared")
        metadata_lines = [
            json.loads(line)
            for line in (out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [row["file_name"] for row in metadata_lines] == ["current.png"]


class TestSourceCaptionSurvivesEndToEnd:
    def test_source_caption_reaches_metadata_jsonl_at_mix_ratio_0(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("uuid-1", "a.png", "a dense VLM description of a login screen", "login screen")],
        )
        out = sync_polish(model_data, tmp_path / "prepared", caption_mix_ratio=0.0)

        rec = json.loads((out / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "login screen"

    def test_dense_caption_reaches_metadata_jsonl_at_mix_ratio_1(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("uuid-1", "a.png", "a dense VLM description of a login screen", "login screen")],
        )
        out = sync_polish(model_data, tmp_path / "prepared", caption_mix_ratio=1.0)

        rec = json.loads((out / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "a dense VLM description of a login screen"

    def test_record_with_no_source_caption_is_unaffected(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("uuid-1", "a.png", "dense only, no source label available")],
        )
        out = sync_polish(model_data, tmp_path / "prepared", caption_mix_ratio=0.0)

        rec = json.loads((out / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "dense only, no source label available"


class TestSyncPolishQwen:
    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        from krivex_training.data_forge_bridge.sync_polish_qwen import sync as sync_qwen
        with pytest.raises(FileNotFoundError, match="Qwen-Image-2.1 pair manifest not found"):
            sync_qwen(tmp_path / "model_data", tmp_path / "prepared")

    def test_sync_qwen_materializes_valid_pairs(self, tmp_path):
        import hashlib
        from krivex_training.data_forge_bridge.sync_polish_qwen import sync as sync_qwen

        model_dir = tmp_path / "model_data" / "polish_qwen_image21"
        cond_dir = model_dir / "conditions"
        target_dir = model_dir / "targets"
        cond_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)

        cond_path = cond_dir / "cond-1.png"
        target_path = target_dir / "target-1.png"
        Image.new("RGB", (16, 16), color="black").save(cond_path)
        Image.new("RGB", (16, 16), color="white").save(target_path)

        cond_sha = hashlib.sha256(cond_path.read_bytes()).hexdigest()
        target_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()

        manifest_line = json.dumps({
            "record_id": "pair-001",
            "condition_image": "conditions/cond-1.png",
            "target_image": "targets/target-1.png",
            "caption": "a crisp UI layout",
            "split": "train",
            "source_dataset": "figma2code",
            "source_provenance": {"repo_id": "test/figma"},
            "condition_sha256": cond_sha,
            "target_sha256": target_sha,
        })
        (model_dir / "manifest.jsonl").write_text(manifest_line + "\n", encoding="utf-8")

        prepared_dir = sync_qwen(tmp_path / "model_data", tmp_path / "prepared")
        assert (prepared_dir / "metadata.jsonl").is_file()
        metadata_lines = [json.loads(line) for line in (prepared_dir / "metadata.jsonl").read_text().splitlines() if line.strip()]
        assert len(metadata_lines) == 1
        assert metadata_lines[0]["pair_id"] == "pair-001"
        assert metadata_lines[0]["caption"] == "a crisp UI layout"

