from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from krisna_training.polish.prepare_dataset import count_prepared, load_captions, prepare
from krisna_training.polish.dataset_prep import prepare as legacy_prepare


def test_prepare_with_dict(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="blue")
    im.save(img_dir / "card.png")

    out_dir = tmp_path / "out"
    prepare(img_dir, out_dir, captions={"card.png": "a blue UI card"})

    meta = out_dir / "metadata.jsonl"
    assert meta.exists()
    assert count_prepared(out_dir) == 1
    rec = json.loads(meta.read_text(encoding="utf-8").strip())
    assert rec["file_name"] == "card.png"
    assert rec["text"] == "a blue UI card"


def test_prepare_with_jsonl(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="red")
    im.save(img_dir / "nav.png")

    jsonl_path = tmp_path / "captions.jsonl"
    jsonl_path.write_text(json.dumps({"image_filename": "nav.png", "caption": "navigation bar"}) + "\n")

    out_dir = tmp_path / "out"
    prepare(img_dir, out_dir, captions=jsonl_path)

    meta = out_dir / "metadata.jsonl"
    assert meta.exists()
    assert count_prepared(out_dir) == 1
    rec = json.loads(meta.read_text(encoding="utf-8").strip())
    assert rec["file_name"] == "nav.png"
    assert rec["text"] == "navigation bar"


def test_legacy_wrapper_works(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="green")
    im.save(img_dir / "icon.png")

    out_dir = tmp_path / "out"
    legacy_prepare(img_dir, out_dir, captions={"icon.png": "an icon"})
    assert count_prepared(out_dir) == 1


class TestSourceCaptionMixing:
    """UPGRADE (docs/review/34_model_review_3_polish_default.md):
    source_caption was exported by data-forge's _export_zimage
    specifically for this purpose, but never actually used anywhere
    until now — see prepare_dataset.py's module docstring for the full
    Betker et al. 2023 reasoning this mirrors from the Sketch tier."""

    def _make_jsonl_with_source_captions(self, tmp_path, records):
        jsonl_path = tmp_path / "captions.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        return jsonl_path

    def test_load_captions_returns_both_dense_and_source(self, tmp_path):
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, [
            {"image_filename": "a.png", "caption": "a dense VLM description",
             "source_caption": "short label"},
        ])
        dense, source = load_captions(jsonl_path)
        assert dense["a.png"] == "a dense VLM description"
        assert source["a.png"] == "short label"

    def test_load_captions_source_missing_is_simply_absent(self, tmp_path):
        """Records with no source_caption (e.g. no matching source-dataset
        label) must not appear in the source dict at all, not as None or
        empty string — prepare()'s mixing logic relies on this to
        correctly skip mixing for those records."""
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, [
            {"image_filename": "a.png", "caption": "dense only", "source_caption": None},
        ])
        dense, source = load_captions(jsonl_path)
        assert dense["a.png"] == "dense only"
        assert "a.png" not in source

    def test_plain_dict_input_has_no_source_captions(self, tmp_path):
        """A plain {filename: caption} dict (the simplest calling
        convention) structurally can't carry source_caption — must
        degrade to "always use dense caption", not crash."""
        dense, source = load_captions({"a.png": "x"})
        assert dense == {"a.png": "x"}
        assert source == {}

    def test_mix_ratio_1_0_always_uses_dense_caption(self, tmp_path):
        img_dir = tmp_path / "src"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "a.png")
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, [
            {"image_filename": "a.png", "caption": "DENSE", "source_caption": "SHORT"},
        ])

        out_dir = tmp_path / "out"
        prepare(img_dir, out_dir, captions=jsonl_path, caption_mix_ratio=1.0, seed=0)

        rec = json.loads((out_dir / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "DENSE"

    def test_mix_ratio_0_0_always_uses_source_caption_when_available(self, tmp_path):
        img_dir = tmp_path / "src"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "a.png")
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, [
            {"image_filename": "a.png", "caption": "DENSE", "source_caption": "SHORT"},
        ])

        out_dir = tmp_path / "out"
        prepare(img_dir, out_dir, captions=jsonl_path, caption_mix_ratio=0.0, seed=0)

        rec = json.loads((out_dir / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "SHORT"

    def test_no_source_caption_available_always_uses_dense_even_at_ratio_0(self, tmp_path):
        """A record with no source_caption at all must always fall back
        to the dense caption, regardless of caption_mix_ratio — there is
        nothing to mix in."""
        img_dir = tmp_path / "src"
        img_dir.mkdir()
        Image.new("RGB", (64, 64)).save(img_dir / "a.png")
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, [
            {"image_filename": "a.png", "caption": "DENSE", "source_caption": None},
        ])

        out_dir = tmp_path / "out"
        prepare(img_dir, out_dir, captions=jsonl_path, caption_mix_ratio=0.0, seed=0)

        rec = json.loads((out_dir / "metadata.jsonl").read_text().strip())
        assert rec["text"] == "DENSE"

    def test_default_ratio_matches_sketch_tier_default(self):
        """Must stay in sync with SketchTokenDataset's default (0.95) —
        both trace back to the SAME Betker et al. citation; a silent
        divergence between the two would be exactly the kind of drift
        this project's review has repeatedly caught elsewhere (e.g. the
        stale P2a caption_mix_ratio default found in an earlier pass)."""
        import inspect

        sig = inspect.signature(prepare)
        assert sig.parameters["caption_mix_ratio"].default == 0.95

    def test_seed_makes_mixing_reproducible(self, tmp_path):
        img_dir = tmp_path / "src"
        img_dir.mkdir()
        for i in range(20):
            Image.new("RGB", (64, 64)).save(img_dir / f"img_{i}.png")
        records = [
            {"image_filename": f"img_{i}.png", "caption": f"DENSE_{i}", "source_caption": f"SHORT_{i}"}
            for i in range(20)
        ]
        jsonl_path = self._make_jsonl_with_source_captions(tmp_path, records)

        out_dir_1 = tmp_path / "out1"
        out_dir_2 = tmp_path / "out2"
        prepare(img_dir, out_dir_1, captions=jsonl_path, caption_mix_ratio=0.5, seed=42)
        prepare(img_dir, out_dir_2, captions=jsonl_path, caption_mix_ratio=0.5, seed=42)

        texts_1 = [json.loads(l)["text"] for l in (out_dir_1 / "metadata.jsonl").read_text().strip().split("\n")]
        texts_2 = [json.loads(l)["text"] for l in (out_dir_2 / "metadata.jsonl").read_text().strip().split("\n")]
        assert texts_1 == texts_2
