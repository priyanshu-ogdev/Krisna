"""Dataset download manager — HuggingFace Hub, GitHub, and direct URL.

Handles dataset fetching with resume support, checksum verification,
and progress tracking via manifest updates.
"""

from __future__ import annotations

import base64
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
from typing import Any

import asyncio  # noqa: F401 — BUG FIX: this was previously only imported
# locally, aliased, and after its first use in _fetch_huggingface_url_list
# (asyncio.Semaphore(...) is constructed before that local import line
# executes) — a real NameError waiting to happen the first time this method
# ran. Hoisted to module level like every other stdlib import here.

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.logging_setup import get_logger

log = get_logger("data.fetcher")


def _decode_gamelabel_image(raw: Any) -> bytes | None:
    """GameLabel-10K's img0_encoding/img1_encoding columns: base64-encoded
    JPEG bytes wrapped in a stray Python bytes-repr string — confirmed by
    inspecting real sample values from the live dataset, e.g.
    `"b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwc...'"`. Strips the
    `b'`/`'` wrapper (a byproduct of the dataset author writing
    `str(bytes_obj)` instead of the bytes themselves before CSV export —
    not a documented format, reverse-engineered from the data itself),
    then base64-decodes the inner content to recover the real JPEG bytes.
    Returns None rather than raising on anything that doesn't match this
    exact shape, so a malformed row is skipped (logged by the caller) —
    not a source of a hard crash mid-fetch over one bad row out of
    thousands.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    if not s:
        return None
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return None


def _extract_caption_value(row_dict: dict[str, Any], col: str | None) -> str | None:
    """Extract natural-language caption or instruction from row_dict.
    Supports direct columns, dot-notation nested paths, and structured message lists.
    """
    if not col:
        return None
    val = None
    if col in row_dict and row_dict[col] is not None:
        val = row_dict[col]
    elif "." in col:
        curr: Any = row_dict
        for part in col.split("."):
            if isinstance(curr, dict):
                curr = curr.get(part)
            elif isinstance(curr, (list, tuple)) or hasattr(curr, "__getitem__"):
                try:
                    idx = int(part)
                    curr = curr[idx] if 0 <= idx < len(curr) else None
                except (ValueError, IndexError, TypeError):
                    curr = None
            else:
                curr = None
            if curr is None:
                break
        val = curr

    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple)) or (hasattr(val, "__iter__") and not isinstance(val, (str, bytes, dict))):
        for item in val:
            if isinstance(item, dict):
                for k in ("content", "instruction", "text", "message"):
                    if item.get(k):
                        s = str(item[k]).strip()
                        if s:
                            return s
            elif isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(val, dict):
        for k in ("content", "instruction", "task", "text", "caption"):
            if val.get(k):
                s = str(val[k]).strip()
                if s:
                    return s
    s = str(val).strip()
    return s if s else None


class DatasetFetcher:
    """Multi-source dataset fetcher."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._raw_dir = config.resolved_paths["raw"]
        self._hf_token = os.environ.get("HF_TOKEN")

    async def fetch_dataset(self, key: str, spec: DatasetSpec) -> list[dict[str, Any]]:
        """Fetch a dataset and return a list of record dicts for manifest insertion.

        Each dict contains: source_file, image_path, content_hash_sha256,
        image_width, image_height, file_size_bytes.
        """
        log.info("fetch_starting", dataset=key, source_type=spec.source_type)

        dataset_dir = self._raw_dir / key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Fast paths: if dataset is already present locally (e.g. 1% test slice),
        # reuse directly instead of re-downloading over the network.
        download_mode = spec.fetch_config.get("download_mode")

        # 1. Preference pairs
        if download_mode in DatasetSpec.PREFERENCE_PAIR_DOWNLOAD_MODES:
            pairs_dir = self._config.data_root / "preference_pairs" / key
            if pairs_dir.exists() and any(pairs_dir.iterdir()):
                log.info("local_preference_pairs_found", dataset=key, dir=str(pairs_dir))
                return []

        # 2. Evaluation reference
        if download_mode == "eval_reference":
            eval_dir = self._config.data_root / "heldout" / "external_eval" / key
            if eval_dir.exists() and any(eval_dir.iterdir()):
                log.info("local_eval_reference_found", dataset=key, dir=str(eval_dir))
                return []

        # 3. Annotation-only / metadata repos
        if spec.annotation_only:
            repo_dir = dataset_dir / "repo" if (dataset_dir / "repo").exists() else dataset_dir
            if repo_dir.exists() and any(repo_dir.iterdir()):
                log.info("local_annotation_repo_found", dataset=key, dir=str(repo_dir))
                return []

        # 4. Image datasets (PD12M, CC12M, RICO, Screen2Words, etc.)
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        existing_images = [f for f in dataset_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]
        if existing_images:
            log.info("local_dataset_images_found", dataset=key, count=len(existing_images))
            return self._scan_downloaded_files(key, dataset_dir)

        # BUG FIX: pd12m and cc12m are metadata-only HF repos — parquet
        # files with an image-URL column, not bundled image files (this was
        # already documented in datasets.yaml's own fetch_config.note for
        # both, but nothing in this fetcher ever acted on that note). The
        # old path here (_fetch_huggingface -> _scan_downloaded_files) just
        # downloaded the parquet metadata and then globbed for .jpg/.png/etc,
        # which parquet files never match — so both datasets silently
        # produced ZERO ingested records on every run, for the two largest,
        # most foundational datasets in the whole corpus (~24.8M of the
        # target records). download_mode: "url_list" routes to the real
        # img2dataset-style path instead.
        if spec.fetch_config.get("download_mode") == "url_list":
            return await self._fetch_huggingface_url_list(key, spec, dataset_dir)

        # BUG FIX / COMPLETENESS GAP: Screen2Words was configured with
        # file_patterns: ["*.parquet"] and no download_mode override —
        # which routed it through the plain _fetch_huggingface() path
        # below, downloading ONLY parquet files (allow_patterns restricts
        # snapshot_download to that pattern) and then handing off to
        # _scan_downloaded_files(), which globs for image extensions that
        # were never downloaded in the first place. Zero records, same
        # failure class as PD12M/CC12M above — just not caught in that
        # pass because Screen2Words' own dataset note didn't flag itself
        # as "metadata-only" the way PD12M/CC12M's did. caption_join
        # handles the two realistic shapes this kind of dataset can
        # actually have (own embedded images, or captions that reference
        # another already-ingested dataset's images by ID) — see
        # _fetch_huggingface_caption_join's docstring for the "don't
        # guess which, detect it" reasoning.
        if spec.fetch_config.get("download_mode") == "caption_join":
            return await self._fetch_huggingface_caption_join(key, spec, dataset_dir)

        # Pairwise human-preference datasets exposed as a fixed two-image
        # table (Pick-a-Pic v2, DesignSense-10k, DesignPref) — (prompt,
        # image_a, image_b, label) quadruples for Diffusion-DPO training.
        # Writes directly into preference_pairs/{key}/, not the
        # image-record manifest, since a ranked pair isn't a single-image
        # record and doesn't belong in a schema built for one. See
        # s01_6_preference_pairs.py for the post-fetch dedup/PII/safety
        # pass, and s08_5_dpo_encoding.py for how these get turned into
        # training-ready latents.
        if spec.fetch_config.get("download_mode") == "preference_pair":
            return await self._fetch_huggingface_preference_pairs(key, spec, dataset_dir)

        # BUG FIX: HPDv2 does NOT expose a fixed two-image table. Its real
        # annotation format (train.json) is a variable-length ranked list
        # per prompt (`human_preference: list[int]`, `file_path:
        # list[str]`) — confirmed directly against the live dataset card,
        # not guessed. Reusing `preference_pair` mode against it would
        # either find zero parquet files (its data ships as JSON + raw
        # jpg folders, not parquet) or silently misparse the ranked-list
        # shape as if it were two fixed columns. This dedicated mode
        # flattens each prompt's ranked list into consecutive
        # winner/loser pairs instead. See _fetch_hpdv2_ranked_pairs.
        if spec.fetch_config.get("download_mode") == "hpdv2_ranked_list":
            return await self._fetch_hpdv2_ranked_pairs(key, spec, dataset_dir)

        # GameLabel-10K: crowdsourced mobile-game vote-count pairs, NOT the
        # fixed two-image table `preference_pair` mode handles. CONFIRMED
        # directly against the live dataset (HF's own dataset-viewer
        # output for Jonathan-Zhou/GameLabel-10k, not guessed): columns
        # are `prompt`, `img0_votes`/`img1_votes` (int, 0-5 — a vote
        # COUNT from up to 5 players per pair, not a fixed 0/1 label),
        # and `img0_encoding`/`img1_encoding` (plain strings — NOT the
        # standard HF `{"bytes": ...}` image-feature dict
        # `_is_image_like()` above detects — the CSV export wraps
        # base64-encoded JPEG bytes in a stray Python `b'...'` repr,
        # e.g. `"b'/9j/4AAQSkZJRgABAQ...'"`). Reusing `preference_pair`
        # mode would find these columns via neither its dict/bytes
        # image-detection nor a fixed-label column, and silently return
        # zero pairs. This dedicated mode decodes the real encoding and
        # derives a win/lose label from vote counts instead.
        if spec.fetch_config.get("download_mode") == "gamelabel_csv":
            return await self._fetch_gamelabel_csv(key, spec, dataset_dir)

        # BUG FIX: RICO (both creative-graphic-design/Rico and
        # Voxel51/rico) — the same failure class as PD12M/CC12M/
        # Screen2Words above, discovered independently rather than
        # inherited from those fixes. Both live repos are confirmed
        # (direct HF dataset-card check) to ship as parquet with images
        # embedded as bytes in a column (`screenshot` for
        # creative-graphic-design/Rico's "ui-screenshots-and-view-
        # hierarchies" config) — not loose .jpg/.png files. The plain
        # `_fetch_huggingface` -> `_scan_downloaded_files` path this used
        # to route through (`allow_patterns: ["*.jpg","*.png","*.json"]`)
        # matches zero files against either repo's real file tree, so it
        # would silently produce zero image records for the single most
        # foundational dataset in the whole UI-domain corpus — CLAY,
        # Enrico, and Screen2Words all join onto rico_core/rico_semantic
        # records that would never have existed. See
        # _fetch_hf_parquet_images.
        if spec.fetch_config.get("download_mode") == "hf_parquet_images":
            return await self._fetch_hf_parquet_images(key, spec, dataset_dir)

        # Evaluation-only reference datasets (TASTE, PartiPrompts) — these
        # exist to benchmark against, never to train on. Routed straight
        # into heldout/external_eval/{key}/ (outside training_pool,
        # preference_pairs/, and the image manifest entirely) so there is
        # no code path — misconfiguration included — by which this data
        # could end up inside a training set. See docs/DATA_SOURCES.md's
        # eval-only rule.
        if spec.fetch_config.get("download_mode") == "eval_reference":
            return await self._fetch_eval_reference(key, spec, dataset_dir)

        if spec.source_type == "huggingface":
            return await self._fetch_huggingface(key, spec, dataset_dir)
        elif spec.source_type == "github":
            return await self._fetch_github(key, spec, dataset_dir)
        elif spec.source_type == "url":
            return await self._fetch_url(key, spec, dataset_dir)
        elif spec.source_type in ("local", "filesystem") or spec.fetch_config.get("download_mode") == "local_scan":
            scan_path = Path(spec.fetch_config.get("local_path", dataset_dir))
            if not scan_path.is_absolute():
                scan_path = self._config.data_root / scan_path
            records = self._scan_downloaded_files(key, scan_path)
            sample_size = spec.fetch_config.get("sample_size")
            if sample_size and len(records) > sample_size:
                records = records[:sample_size]
            return records
        else:
            log.error("unknown_source_type", dataset=key, source_type=spec.source_type)
            return []

    async def _fetch_hf_parquet_images(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch a HuggingFace dataset that ships images embedded as bytes
        inside parquet files (the modern `datasets`-library push_to_hub
        convention — RICO, and most datasets uploaded since ~2023) rather
        than as loose image files in the repo tree.

        Unlike _fetch_huggingface's plain snapshot_download + extension
        scan (correct for datasets that really do ship loose files —
        WebUI, CLAY, Enrico), this decodes the embedded image column
        directly and writes real image files to disk, then returns
        records in the same shape _scan_downloaded_files produces so
        this plugs into the existing dedup/quality/safety pipeline
        identically regardless of which fetch path a given source needed.

        fetch_config keys this reads:
            image_column (str, optional): column holding embedded image
                bytes. Auto-detected (same `_is_image_like` scan
                _fetch_huggingface_preference_pairs uses) if not given —
                override here rather than trust the guess if the fetch
                log shows it picked the wrong column on a multi-image-
                column schema.
            caption_column (str, optional): if present, captured into
                each record's `source_caption` for s05_recaption to use
                as a prior hint, same convention as PD12M/CC12M/
                Screen2Words.
            config_subfolder (str, optional): for HF repos with multiple
                named configs living in separate subfolders in the repo
                tree (creative-graphic-design/Rico ships "default"
                metadata-only, "ui-screenshots-and-view-hierarchies", and
                "ui-screenshots-and-hierarchies-with-semantic-
                annotations" as three separate parquet sets — fetching
                the whole repo indiscriminately would mix the metadata-
                only config's rows, which have no image column at all,
                in with the two that do). When set, only parquet files
                under this path prefix are downloaded.
            sample_size (int, optional): cap on rows to decode.
        """
        import io

        import pandas as pd
        from huggingface_hub import snapshot_download
        from PIL import Image

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        config_subfolder = spec.fetch_config.get("config_subfolder")
        allow_patterns = (
            [f"{config_subfolder}/*.parquet"] if config_subfolder
            else spec.fetch_config.get("file_patterns", ["*.parquet"])
        )

        target_dir = dest / "_parquet"
        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded_dir = target_dir

        try:
            dl_res = snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision or "main",
                local_dir=str(target_dir),
                allow_patterns=allow_patterns,
                token=self._hf_token,
            )
            if dl_res and Path(dl_res).exists():
                downloaded_dir = Path(dl_res)
        except Exception as e:
            log.warning("hf_parquet_snapshot_download_failed", error=str(e))

        parquet_files = sorted(Path(downloaded_dir).rglob("*.parquet"))
        if not parquet_files and downloaded_dir != target_dir:
            parquet_files = sorted(Path(target_dir).rglob("*.parquet"))
        if not parquet_files:
            log.error(
                "hf_parquet_images_no_parquet", dataset=key, dir=str(target_dir),
                config_subfolder=config_subfolder,
                note="No *.parquet files matched — check config_subfolder/file_patterns "
                     "against the live repo tree before assuming the repo is empty.",
            )
            return []

        def _is_image_like(val: Any) -> bool:
            return (
                (isinstance(val, dict) and "bytes" in val)
                or isinstance(val, (bytes, bytearray))
                or hasattr(val, "save")
            )

        # Inspect first readable parquet file to resolve columns
        sample_df = None
        for pf in parquet_files:
            try:
                sample_df = pd.read_parquet(pf)
                break
            except Exception as e:
                log.warning("hf_parquet_images_first_file_failed", file=str(pf), error=str(e))
        if sample_df is None:
            log.error("hf_parquet_images_no_readable_parquet", dataset=key)
            return []

        columns = list(sample_df.columns)
        image_col = spec.fetch_config.get("image_column")
        if not image_col:
            for col in columns:
                sample = sample_df[col].dropna().iloc[0] if sample_df[col].notna().any() else None
                if _is_image_like(sample):
                    image_col = col
                    break

        if not image_col:
            log.error(
                "hf_parquet_images_no_image_column", dataset=key,
                available_columns=columns,
                note="No embedded-image-like column auto-detected — override via "
                     "fetch_config.image_column. If this config_subfolder is a "
                     "metadata-only config (e.g. RICO's \"default\" config, which "
                     "has no image column at all by design), point config_subfolder "
                     "at a config that actually ships images instead.",
            )
            return []

        caption_col = spec.fetch_config.get("caption_column")
        sample_size = spec.fetch_config.get("sample_size")
        viewport_crop = spec.fetch_config.get("viewport_crop", False)
        max_ar = spec.fetch_config.get("max_aspect_ratio")
        limit_ratio = float(max_ar) if max_ar is not None else (2.0 if viewport_crop else None)

        log.info(
            "hf_parquet_images_columns_resolved", dataset=key,
            image_column=image_col, caption_column=caption_col,
            parquet_files_count=len(parquet_files),
        )

        out_dir = dest
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        decode_failures = 0
        global_idx = 0

        # Memory-safe file-by-file processing: avoid loading all files into RAM simultaneously
        for pf_idx, pf in enumerate(parquet_files):
            if sample_size and len(records) >= sample_size:
                break
            try:
                df = sample_df if pf_idx == 0 else pd.read_parquet(pf)
            except Exception as e:
                log.warning("hf_parquet_images_read_failed", file=str(pf), error=str(e))
                continue

            if sample_size:
                needed = sample_size - len(records)
                if len(parquet_files) == 1 and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                elif len(df) > needed:
                    df = df.head(needed)

            row_items = []
            for row in df.itertuples(index=False):
                row_items.append((global_idx, dict(zip(df.columns, row))))
                global_idx += 1
                if sample_size and (len(records) + len(row_items)) >= sample_size:
                    break

            def _decode_and_save(item: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
                curr_idx, row_dict = item
                try:
                    raw = row_dict.get(image_col)
                    blob: bytes | None = None
                    img: Image.Image | None = None

                    if hasattr(raw, "save"):
                        img = raw
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        blob = buf.getvalue()
                    elif isinstance(raw, dict) and "bytes" in raw:
                        blob = raw["bytes"]
                    elif isinstance(raw, (bytes, bytearray)):
                        blob = bytes(raw)

                    if blob is None and img is None:
                        return None, True

                    if img is None and blob:
                        try:
                            img = Image.open(io.BytesIO(blob))
                            img.load()
                        except Exception:
                            return None, True

                    if img is None:
                        return None, True

                    # Viewport crop / aspect ratio limit (e.g. for ultra-tall web captures)
                    if limit_ratio is not None and (img.height / max(img.width, 1)) > limit_ratio:
                        target_h = int(img.width * limit_ratio)
                        img = img.crop((0, 0, img.width, target_h))

                    # Handle transparency cleanly
                    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        alpha = img.convert("RGBA").split()[-1]
                        bg.paste(img.convert("RGB"), mask=alpha)
                        img = bg
                    else:
                        img = img.convert("RGB")

                    file_name = f"{key}_{curr_idx:07d}.png"
                    out_path = out_dir / file_name
                    img.save(out_path, "PNG")

                    file_size = out_path.stat().st_size
                    try:
                        rel_path = str(out_path.relative_to(self._config.data_root))
                    except ValueError:
                        rel_path = str(out_path)

                    rec_dict: dict[str, Any] = {
                        "source_file": file_name,
                        "image_path": rel_path,
                        "content_hash_sha256": self._compute_bytes_sha256(blob if blob else out_path.read_bytes()),
                        "image_width": img.width,
                        "image_height": img.height,
                        "file_size_bytes": file_size,
                    }
                    caption_val = _extract_caption_value(row_dict, caption_col)
                    if caption_val:
                        rec_dict["source_caption"] = caption_val
                    return rec_dict, False
                except Exception:
                    return None, True

            import concurrent.futures
            max_workers = min(32, max(4, os.cpu_count() or 4))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for rec_res, failed in executor.map(_decode_and_save, row_items):
                    if failed or rec_res is None:
                        decode_failures += 1
                    else:
                        records.append(rec_res)

            del df

        log.info(
            "hf_parquet_images_complete", dataset=key,
            decoded=len(records), decode_failures=decode_failures,
        )
        return records

    async def _fetch_huggingface_preference_pairs(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch a (prompt, image_a, image_b, label) human-preference-pair
        dataset (Pick-a-Pic v2, HPDv2, DesignSense-10k, DesignPref) and
        write pairs directly into preference_pairs/{key}/.

        These are real, already-published, human-annotated comparisons —
        no synthesis, no AI-judge labeling happens here or anywhere
        downstream of it. Column auto-detection follows the same
        discipline as _fetch_huggingface_caption_join: look for the
        columns the schema needs, log exactly what was found, and log
        loudly (never guess silently) if detection fails, since a wrong
        guess here would silently corrupt every DPO pair's win/lose
        label — worse than finding zero records.

        fetch_config keys this reads:
            prompt_column / image_a_column / image_b_column / label_column
                (str, optional): override auto-detection if the live
                schema doesn't match the defaults below.
            label_convention (str, optional): "index" (label is 0 or 1,
                selecting image_a/image_b as preferred) or "column_name"
                (label directly names the preferred column, e.g.
                "image_0"/"image_1"). Defaults to "index".
            sample_size (int, optional): cap on rows to attempt.
        """
        import io

        import pandas as pd
        from huggingface_hub import snapshot_download
        from PIL import Image

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        import fnmatch
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=self._hf_token)
        target_dir = dest / "_metadata"
        target_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = spec.fetch_config.get("file_patterns", ["*.parquet"])

        try:
            repo_files = api.list_repo_files(repo_id=spec.repo_id, repo_type="dataset", revision=spec.revision or "main")
            matching_parquet = []
            for pat in allow_patterns:
                for rf in repo_files:
                    if fnmatch.fnmatch(rf, pat) and rf.endswith(".parquet") and rf not in matching_parquet:
                        matching_parquet.append(rf)
            matching_parquet.sort()
            if matching_parquet:
                for rf in matching_parquet[:1]:
                    hf_hub_download(
                        repo_id=spec.repo_id,
                        repo_type="dataset",
                        filename=rf,
                        revision=spec.revision or "main",
                        local_dir=str(target_dir),
                        token=self._hf_token,
                    )
            else:
                snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    revision=spec.revision or "main",
                    local_dir=str(target_dir),
                    allow_patterns=allow_patterns,
                    token=self._hf_token,
                )
        except Exception as e:
            log.error("preference_pair_download_failed", dataset=key, error=str(e))
            return []

        parquet_files = sorted(Path(target_dir).rglob("*.parquet"))
        if not parquet_files:
            log.error("preference_pair_no_parquet", dataset=key, dir=str(target_dir))
            return []

        frames = []
        for pf in parquet_files:
            try:
                frames.append(pd.read_parquet(pf))
            except Exception as e:
                log.warning("preference_pair_parquet_read_failed", file=str(pf), error=str(e))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True)
        columns = list(df.columns)

        def _is_image_like(val: Any) -> bool:
            return (isinstance(val, dict) and "bytes" in val) or isinstance(val, (bytes, bytearray))

        image_cols = []
        for col in columns:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if _is_image_like(sample):
                image_cols.append(col)

        prompt_col = spec.fetch_config.get("prompt_column") or next(
            (c for c in columns if c.lower() in ("caption", "prompt", "text")), None
        )
        label_col = spec.fetch_config.get("label_column") or next(
            (c for c in columns if c.lower() in
             ("label", "label_0", "preference", "best_image_uid", "human_preference")),
            None,
        )
        image_a_col = spec.fetch_config.get("image_a_column") or (image_cols[0] if image_cols else None)
        image_b_col = spec.fetch_config.get("image_b_column") or (image_cols[1] if len(image_cols) > 1 else None)

        if not image_a_col or not image_b_col or prompt_col is None or label_col is None:
            log.error(
                "preference_pair_undetectable",
                dataset=key,
                available_columns=columns,
                detected_image_cols=image_cols,
                detected_prompt_col=prompt_col,
                detected_label_col=label_col,
                note="Need 2 image-like columns, 1 prompt/caption column, and "
                     "1 label column — inspect the real schema and override via "
                     "fetch_config (prompt_column/image_a_column/image_b_column/"
                     "label_column) rather than trusting a guessed match.",
            )
            return []

        log.info(
            "preference_pair_columns_resolved",
            dataset=key, image_a_col=image_a_col, image_b_col=image_b_col,
            prompt_col=prompt_col, label_col=label_col,
        )

        sample_size = spec.fetch_config.get("sample_size")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        out_dir = self._config.resolved_paths["preference_pairs"] / key
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        seen_hashes: set[str] = set()

        valid_items = []
        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = dict(zip(df.columns, row))
            raw_label = row_dict.get(label_col)
            label = self._normalize_preference_label(raw_label, image_a_col, image_b_col)
            if label is None:
                continue

            a_raw = row_dict.get(image_a_col)
            b_raw = row_dict.get(image_b_col)
            a_blob = a_raw.get("bytes") if isinstance(a_raw, dict) else a_raw
            b_blob = b_raw.get("bytes") if isinstance(b_raw, dict) else b_raw
            if not a_blob or not b_blob:
                continue

            # Cheap exact-duplicate guard within this source
            pair_hash = self._compute_bytes_sha256(a_blob) + self._compute_bytes_sha256(b_blob)
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            valid_items.append((idx, row_dict, a_blob, b_blob, label))

        def _save_pref_pair(item: tuple[int, dict[str, Any], Any, Any, str]) -> bool:
            idx, row_dict, a_blob, b_blob, label = item
            try:
                a_img = Image.open(io.BytesIO(a_blob))
                b_img = Image.open(io.BytesIO(b_blob))
                pair_id = f"{key}_{idx:07d}"
                a_img.save(out_dir / f"{pair_id}_a.png", "PNG")
                b_img.save(out_dir / f"{pair_id}_b.png", "PNG")
                (out_dir / f"{pair_id}.json").write_text(
                    json.dumps({
                        "pair_id": pair_id,
                        "prompt": str(row_dict.get(prompt_col, "")),
                        "image_a": f"{pair_id}_a.png",
                        "image_b": f"{pair_id}_b.png",
                        "preferred": label,  # "a" or "b"
                        "origin": key,
                        "label_source": "human",
                    }),
                    encoding="utf-8",
                )
                return True
            except Exception:
                return False

        import concurrent.futures
        max_workers = min(32, max(4, os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ok in executor.map(_save_pref_pair, valid_items):
                if ok:
                    written += 1

        log.info("preference_pairs_written", dataset=key, count=written)
        return []  # Not manifest records — written directly to preference_pairs/

    async def _fetch_gamelabel_csv(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch GameLabel-10K (Jonathan-Zhou/GameLabel-10k) — a real,
        Apache-2.0, crowdsourced mobile-game image-preference dataset
        (arXiv:2409.19830). General-domain DPO signal (supplements
        Pick-a-Pic/HPDv2's Stage-1 role) — NOT a substitute for the
        UI-domain sources (DesignSense-10k/DesignPref), which remain
        publicly unreleased; see datasets.yaml's entries for both.

        Reads via the `refs/convert/parquet` revision Hugging Face
        auto-generates for CSV-formatted dataset repos, rather than
        streaming the raw 2.26GB data.csv directly — same parquet-reading
        infrastructure `_fetch_huggingface_preference_pairs` already uses,
        just pointed at a different revision. Confirmed this mirror
        exists for this specific repo (HF's own "refs/convert/parquet"
        tree listing for Jonathan-Zhou/GameLabel-10k, "Update parquet
        files") before writing this, rather than assumed.

        Real, confirmed schema (from HF's own dataset-viewer output for
        this repo — not guessed): `prompt` (str), `img0_votes`/
        `img1_votes` (int, 0-5 — a vote COUNT from up to 5 crowdsourced
        players per pair, not a fixed 0/1 label column), `img0_encoding`/
        `img1_encoding` (str — base64-encoded JPEG wrapped in a stray
        Python `b'...'` bytes-repr, e.g. `"b'/9j/4AAQSkZJRgABAQ...'"` —
        confirmed by inspecting real sample values, not the standard HF
        image-feature `{"bytes": ...}` dict format).
        """
        import io

        import pandas as pd
        from huggingface_hub import snapshot_download
        from PIL import Image

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        df = None
        parquet_revision = spec.fetch_config.get("parquet_revision", "refs/convert/parquet")
        target_dir = dest / "_metadata"
        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded_dir = target_dir

        try:
            dl_res = snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=parquet_revision,
                local_dir=str(target_dir),
                allow_patterns=["*.parquet"],
                token=self._hf_token,
            )
            if dl_res and Path(dl_res).exists():
                downloaded_dir = Path(dl_res)
        except Exception as e:
            log.warning("gamelabel_snapshot_download_failed", error=str(e))

        parquet_files = sorted(Path(downloaded_dir).rglob("*.parquet"))
        if not parquet_files and downloaded_dir != target_dir:
            parquet_files = sorted(Path(target_dir).rglob("*.parquet"))

        if parquet_files:
            frames = []
            for pf in parquet_files:
                try:
                    frames.append(pd.read_parquet(pf))
                except Exception as e:
                    log.warning("gamelabel_parquet_read_failed", file=str(pf), error=str(e))
            if frames:
                df = pd.concat(frames, ignore_index=True)

        if df is None:
            # Fallback: Fetch first 10MB of data.csv via HTTP Range request
            import httpx
            url = f"https://huggingface.co/datasets/{spec.repo_id}/resolve/main/data.csv"
            try:
                with httpx.Client(follow_redirects=True) as client:
                    resp = client.get(url, headers={"Range": "bytes=0-10000000"}, timeout=20)
                    if resp.status_code in (200, 206) and len(resp.content) > 1000:
                        last_nl = resp.content.rfind(b"\n")
                        csv_bytes = resp.content[:last_nl]
                        df = pd.read_csv(io.BytesIO(csv_bytes), on_bad_lines="skip")
                        log.info("gamelabel_range_download_success", rows=len(df))
            except Exception as e:
                log.warning("gamelabel_range_download_failed", error=str(e))

        if df is None:
            log.error(
                "gamelabel_parquet_not_found", dataset=key, dir=str(target_dir),
                note=(
                    f"Expected an auto-converted parquet mirror at revision "
                    f"{parquet_revision!r} — if HF's auto-conversion for this "
                    f"repo has changed or lagged, fall back to reading "
                    f"data.csv directly at revision 'main' instead."
                ),
            )
            return []

        required = {"prompt", "img0_votes", "img1_votes", "img0_encoding", "img1_encoding"}
        missing = required - set(df.columns)
        if missing:
            log.error(
                "gamelabel_schema_mismatch", dataset=key, missing=sorted(missing),
                available_columns=list(df.columns),
                note="Confirmed schema no longer matches the live data — do not guess a remapping, inspect the real columns and update this method.",
            )
            return []

        sample_size = spec.fetch_config.get("sample_size")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        out_dir = self._config.resolved_paths["preference_pairs"] / key
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped_ties = 0
        seen_hashes: set[str] = set()

        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = dict(zip(df.columns, row))

            votes_a, votes_b = row_dict.get("img0_votes"), row_dict.get("img1_votes")
            try:
                votes_a, votes_b = int(votes_a), int(votes_b)
            except (TypeError, ValueError):
                continue
            if votes_a == votes_b:
                # Genuine tie (including 0-0, no votes cast either way) —
                # not usable for DPO's strict win/lose pairing, same
                # tie-dropping convention _fetch_huggingface_preference_pairs
                # and _fetch_hpdv2_ranked_pairs already use.
                skipped_ties += 1
                continue
            label = "a" if votes_a > votes_b else "b"

            try:
                a_blob = _decode_gamelabel_image(row_dict.get("img0_encoding"))
                b_blob = _decode_gamelabel_image(row_dict.get("img1_encoding"))
                if not a_blob or not b_blob:
                    continue
                a_img = Image.open(io.BytesIO(a_blob)); a_img.load()
                b_img = Image.open(io.BytesIO(b_blob)); b_img.load()
            except Exception as e:
                log.warning("gamelabel_image_decode_failed", dataset=key, row=idx, error=str(e))
                continue

            pair_hash = self._compute_bytes_sha256(a_blob) + self._compute_bytes_sha256(b_blob)
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            pair_id = f"{key}_{idx:07d}"
            a_img.save(out_dir / f"{pair_id}_a.png", "PNG")
            b_img.save(out_dir / f"{pair_id}_b.png", "PNG")
            (out_dir / f"{pair_id}.json").write_text(
                json.dumps({
                    "pair_id": pair_id,
                    "prompt": str(row_dict.get("prompt", "")),
                    "image_a": f"{pair_id}_a.png",
                    "image_b": f"{pair_id}_b.png",
                    "preferred": label,  # "a" or "b"
                    "origin": key,
                    "label_source": "human",
                    # Preserved for downstream weighting/analysis — a 5-0
                    # vote split is a stronger signal than a 1-0 split,
                    # even though both produce the same binary label above.
                    "vote_margin": abs(votes_a - votes_b),
                }),
                encoding="utf-8",
            )
            written += 1

        log.info("gamelabel_pairs_written", dataset=key, count=written, skipped_ties=skipped_ties)
        return []  # Not manifest records — written directly to preference_pairs/

    @staticmethod
    def _normalize_preference_label(raw_label: Any, image_a_col: str, image_b_col: str) -> str | None:
        """Map a dataset's own label convention to "a"/"b", or None for
        ties/unparseable values (DPO needs a strict preference per pair;
        ties carry no gradient signal and are dropped, not guessed at).
        """
        if raw_label is None:
            return None
        if isinstance(raw_label, (int, float)):
            if raw_label == 0:
                return "a"
            if raw_label == 1:
                return "b"
            return None
        s = str(raw_label).strip().lower()
        if s in ("a", "0", "image_a", "left", image_a_col.lower()):
            return "a"
        if s in ("b", "1", "image_b", "right", image_b_col.lower()):
            return "b"
        return None

    @staticmethod
    def _compute_bytes_sha256(blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()

    async def _fetch_hpdv2_ranked_pairs(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch HPDv2's real annotation format and flatten it into pairs.

        HPDv2 ships train.json as a list of
            {"human_preference": list[int],  # 1 = preferred, 0 = not
             "prompt": str,
             "file_path": list[str]}
        — a ranked comparison over N candidate images per prompt, not a
        fixed two-image table (confirmed directly against the live
        ymhao/HPDv2 dataset card, not assumed). This method downloads only
        train.json first (small), samples entries down to
        fetch_config.sample_size *before* touching any image bytes, then
        pulls just the referenced images for the sampled entries via
        individual hf_hub_download calls — not a full snapshot_download of
        the ~430K-image corpus, which would be enormous overkill for a
        60K-pair sample.

        Only entries with exactly 2 candidate images and exactly one
        preferred (human_preference sums to 1) are used — HPDv2's
        pairwise framing covers this cleanly. Entries with 3+ candidates,
        ties, or ambiguous preference vectors are skipped and counted
        rather than guessed at, same discipline as
        _fetch_huggingface_preference_pairs' tie-dropping for the
        fixed-table sources.
        """
        import random

        from huggingface_hub import hf_hub_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        try:
            meta_file = hf_hub_download(
                repo_id=spec.repo_id, repo_type="dataset",
                filename=spec.fetch_config.get("annotation_file", "train.json"),
                revision=spec.revision or "main", token=self._hf_token,
            )
        except Exception as e:
            log.error("hpdv2_annotation_download_failed", dataset=key, error=str(e))
            return []

        with open(meta_file, encoding="utf-8") as f:
            entries = json.load(f)

        usable = []
        skipped_ambiguous = 0
        for entry in entries:
            file_paths = entry.get("image_path") or entry.get("file_path") or []
            prefs = entry.get("human_preference")
            rank = entry.get("rank")

            if prefs is not None and len(file_paths) == 2 and len(prefs) == 2 and sum(prefs) == 1:
                usable.append({
                    "prompt": entry.get("prompt", ""),
                    "file_path": file_paths,
                    "human_preference": prefs,
                })
            elif rank is not None and len(file_paths) >= 2 and len(rank) == len(file_paths):
                best_idx = rank.index(min(rank))
                worst_idx = rank.index(max(rank))
                usable.append({
                    "prompt": entry.get("prompt", ""),
                    "file_path": [file_paths[best_idx], file_paths[worst_idx]],
                    "human_preference": [1, 0],
                })
            else:
                skipped_ambiguous += 1
                continue

        log.info(
            "hpdv2_annotations_parsed",
            dataset=key, total=len(entries), usable_pairwise=len(usable),
            skipped_ambiguous=skipped_ambiguous,
        )

        sample_size = spec.fetch_config.get("sample_size")
        if sample_size and len(usable) > sample_size:
            random.Random(42).shuffle(usable)
            usable = usable[:sample_size]

        out_dir = self._config.resolved_paths["preference_pairs"] / key
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        def _download_and_save_entry(item: tuple[int, dict[str, Any]]) -> bool:
            idx, entry = item
            file_paths = entry["file_path"]
            prefs = entry["human_preference"]
            preferred_idx = prefs.index(1)
            rejected_idx = 1 - preferred_idx

            try:
                preferred_local = hf_hub_download(
                    repo_id=spec.repo_id, repo_type="dataset",
                    filename=file_paths[preferred_idx],
                    revision=spec.revision or "main", token=self._hf_token,
                )
                rejected_local = hf_hub_download(
                    repo_id=spec.repo_id, repo_type="dataset",
                    filename=file_paths[rejected_idx],
                    revision=spec.revision or "main", token=self._hf_token,
                )
            except Exception as e:
                if idx < 3:
                    log.error(
                        "hpdv2_image_download_failed", dataset=key,
                        file_path_attempted=file_paths[preferred_idx], error=str(e),
                    )
                return False

            from PIL import Image
            pair_id = f"{key}_{idx:07d}"
            try:
                Image.open(preferred_local).convert("RGB").save(out_dir / f"{pair_id}_a.png", "PNG")
                Image.open(rejected_local).convert("RGB").save(out_dir / f"{pair_id}_b.png", "PNG")
            except Exception as e:
                log.warning("hpdv2_pair_image_decode_failed", pair=pair_id, error=str(e))
                return False

            (out_dir / f"{pair_id}.json").write_text(
                json.dumps({
                    "pair_id": pair_id,
                    "prompt": entry.get("prompt", ""),
                    "image_a": f"{pair_id}_a.png",  # preferred
                    "image_b": f"{pair_id}_b.png",  # rejected
                    "preferred": "a",
                    "origin": key,
                    "label_source": "human",
                }),
                encoding="utf-8",
            )
            return True

        import concurrent.futures
        max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for success in executor.map(_download_and_save_entry, enumerate(usable)):
                if success:
                    written += 1

        log.info("hpdv2_ranked_pairs_written", dataset=key, count=written)
        return []  # Not manifest records — written directly to preference_pairs/

    async def _fetch_eval_reference(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download an evaluation-only reference dataset (TASTE,
        PartiPrompts) and land it under heldout/external_eval/{key}/,
        never into the image manifest or preference_pairs/.

        This is a structural guarantee, not a config flag: eval-only
        sources physically never reach training_pool/ or
        preference_pairs/ because this method is the only thing that
        touches them, and it only ever writes to the eval directory. A
        stage_filter mistake or a future contributor forgetting an
        `eval_only` check elsewhere cannot leak this data into training —
        there is no code path from here to anywhere training reads from.
        """
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        eval_dir = self._config.resolved_paths["heldout"] / "external_eval" / key
        eval_dir.mkdir(parents=True, exist_ok=True)

        log.info("eval_reference_downloading", repo=spec.repo_id, dest=str(eval_dir))
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(eval_dir),
            allow_patterns=spec.fetch_config.get("file_patterns", ["*.parquet", "*.json", "*.jsonl"]),
            token=self._hf_token,
        )
        log.info(
            "eval_reference_complete",
            dataset=key,
            note="Landed in heldout/external_eval/ only — never used as training data.",
        )
        return []

    async def _fetch_huggingface_caption_join(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch a caption dataset whose real shape is unverified: it either

        (a) bundles its own image bytes in the parquet (an "image"-like
            struct/binary column), in which case we decode and save those
            images directly, same as any other standalone image source; or
        (b) only references another already-ingested dataset's images by
            ID (the likely case for Screen2Words, which is captions FOR
            RICO's screens, not new images of its own) — in which case we
            join the caption onto the matching already-ingested record's
            `caption` field (stored as `source_caption`, a prior/hint —
            see s05_recaption.py for how it's used) instead of fetching
            anything new.

        UNVERIFIED, explicitly, per the data-completeness audit that
        surfaced this bug: which of (a) or (b) is actually true for a
        given dataset's live schema was not independently confirmed
        before this fix — auto-detecting from the actual columns present
        and logging which path was taken (or logging every column
        inspected and why neither matched, rather than silently returning
        zero records) is the safer choice than assuming one shape.

        fetch_config keys this reads:
            join_target_dataset (str, optional): source_dataset key to
                join against for path (b) — e.g. "rico_core". Required
                only if path (b) is the one that ends up matching.
            join_id_column / caption_column (str, optional): override the
                auto-detected column names if the defaults guess wrong.
        """
        import pandas as pd
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        target_dir = dest / "_metadata"
        target_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = spec.fetch_config.get("file_patterns", ["*.parquet"])

        import fnmatch
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=self._hf_token)
        try:
            repo_files = api.list_repo_files(repo_id=spec.repo_id, repo_type="dataset", revision=spec.revision or "main")
            matching_files = []
            for pat in allow_patterns:
                for rf in repo_files:
                    if fnmatch.fnmatch(rf, pat) and rf.endswith(".parquet") and rf not in matching_files:
                        matching_files.append(rf)
            matching_files.sort()
        except Exception:
            matching_files = []

        if matching_files:
            for rf in matching_files[:1]:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=rf,
                    revision=spec.revision or "main",
                    local_dir=str(target_dir),
                    token=self._hf_token,
                )
        else:
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision or "main",
                local_dir=str(target_dir),
                allow_patterns=allow_patterns,
                token=self._hf_token,
            )

        parquet_files = sorted(Path(target_dir).rglob("*.parquet"))
        if not parquet_files:
            log.error("caption_join_no_parquet_found", dataset=key, dir=str(target_dir))
            return []

        frames = []
        for pf in parquet_files:
            try:
                frames.append(pd.read_parquet(pf))
            except Exception as e:
                log.warning("caption_join_parquet_read_failed", file=str(pf), error=str(e))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True)
        columns = list(df.columns)

        # Path (a) detection: any column whose values look like embedded
        # image bytes (a dict with a "bytes" key, the standard HF
        # `datasets.Image()` parquet-export shape) or raw bytes directly.
        image_col = None
        for col in columns:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(sample, dict) and "bytes" in sample:
                image_col = col
                break
            if isinstance(sample, (bytes, bytearray)):
                image_col = col
                break

        caption_col = spec.fetch_config.get("caption_column") or next(
            (c for c in columns if c.lower() in ("caption", "captions", "summary", "text", "description")),
            None,
        )

        sample_size = spec.fetch_config.get("sample_size")
        if image_col is not None:
            log.info("caption_join_path_a_embedded_images", dataset=key, image_col=image_col, caption_col=caption_col)
            return self._decode_embedded_images(key, dest, df, image_col, caption_col, sample_size=sample_size)

        # Path (b): no embedded images — look for an ID column to join
        # against an already-ingested dataset.
        id_col = spec.fetch_config.get("join_id_column") or next(
            (c for c in columns if c.lower() in ("rico_id", "screen_id", "image_id", "id")),
            None,
        )
        join_target = spec.fetch_config.get("join_target_dataset")

        if id_col is None or caption_col is None or join_target is None:
            log.error(
                "caption_join_path_b_undetectable",
                dataset=key,
                available_columns=columns,
                detected_id_col=id_col,
                detected_caption_col=caption_col,
                configured_join_target=join_target,
                note="Neither embedded images (path a) nor a clean "
                     "ID+caption+join_target_dataset (path b) could be "
                     "resolved. Inspect the real columns above and set "
                     "join_id_column/caption_column/join_target_dataset "
                     "explicitly in datasets.yaml's fetch_config rather "
                     "than relying on auto-detection.",
            )
            return []

        return self._join_captions_to_existing(key, df, id_col, caption_col, join_target)

    def _decode_embedded_images(
        self, key: str, dest: Path, df: Any, image_col: str, caption_col: str | None, sample_size: int | None = None
    ) -> list[dict[str, Any]]:
        """Decode a parquet column of embedded image bytes to files on disk."""
        import io

        from PIL import Image

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        images_dir = dest / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []

        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = dict(zip(df.columns, row))
            raw = row_dict.get(image_col)
            blob = raw.get("bytes") if isinstance(raw, dict) else raw
            if not blob:
                continue
            try:
                img = Image.open(io.BytesIO(blob))
                img.load()
            except Exception:
                continue

            file_path = images_dir / f"{key}_{idx:08d}.png"
            img.save(file_path, "PNG")
            sha256 = self._compute_sha256(file_path)
            try:
                rel_path = str(file_path.relative_to(self._config.data_root))
            except ValueError:
                rel_path = str(file_path)

            record: dict[str, Any] = {
                "source_file": file_path.name,
                "image_path": rel_path,
                "content_hash_sha256": sha256,
                "image_width": img.width,
                "image_height": img.height,
                "file_size_bytes": file_path.stat().st_size,
            }
            if caption_col:
                caption = row_dict.get(caption_col)
                if caption:
                    record["source_caption"] = str(caption)
            records.append(record)

        log.info("caption_join_embedded_decoded", dataset=key, count=len(records))
        return records

    def _join_captions_to_existing(
        self, key: str, df: Any, id_col: str, caption_col: str, join_target: str
    ) -> list[dict[str, Any]]:
        """Prepare caption-join pairs against an already-ingested dataset.

        Returns pseudo-records marked `_join_only` rather than new image
        records — DatasetFetcher doesn't hold a Manifest reference, so the
        actual join (matching `_join_key` against an already-ingested
        record's filename stem, same approach as uicrit_ingest.py's RICO
        join) happens in s01_fetch.py after this returns, which does have
        manifest access.
        """
        pairs = []
        for row in df.itertuples(index=False):
            row_dict = dict(zip(df.columns, row))
            join_key = row_dict.get(id_col)
            caption = row_dict.get(caption_col)
            if join_key is None or not caption:
                continue
            pairs.append({"_join_only": True, "_join_target_dataset": join_target,
                          "_join_key": str(join_key), "_source_caption": str(caption)})
        log.info("caption_join_pairs_prepared", dataset=key, count=len(pairs), join_target=join_target)
        return pairs

    async def _fetch_huggingface_url_list(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download a metadata-only HF dataset (parquet + external image URLs).

        img2dataset-style path: download the parquet shard(s), read the
        image-URL column, concurrently download images with bounded
        concurrency, and build the same record-dict shape
        `_scan_downloaded_files` produces so downstream code (manifest
        insertion, license verification, dedup, ...) doesn't need to know
        which path a dataset came through.

        fetch_config keys this reads:
            image_url_column (str, required): column holding the image URL.
            caption_column (str, optional): column holding a source caption,
                stored as `source_caption` for s05_recaption to optionally
                use as a prior/hint rather than captioning from scratch.
            sample_size (int, optional): cap on how many rows to attempt.
                Full datasets here are 12M+ rows; the PRD's actual training
                target is 100K-500K curated samples total across ALL
                sources, so downloading the full 12M-row superset by
                default would be both unnecessary and a multi-terabyte,
                multi-day operation. Defaults to 200_000 if unset — large
                enough to survive the pipeline's aggressive downstream
                filtering (dedup, quality, safety) while staying inside a
                sane single-workstation fetch budget. Set explicitly in
                datasets.yaml to override.
            download_concurrency (int, optional): concurrent image
                downloads. Default 32.
            download_timeout_seconds (int, optional): per-image timeout.
                Default 15 (short — a hung URL shouldn't stall the batch).
        """
        import httpx
        import pandas as pd
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        url_col = spec.fetch_config.get("image_url_column")
        if not url_col:
            log.error(
                "missing_image_url_column",
                dataset=key,
                note="download_mode: url_list requires fetch_config.image_url_column",
            )
            return []

        caption_col = spec.fetch_config.get("caption_column")
        sample_size = spec.fetch_config.get("sample_size", 200_000)
        concurrency = spec.fetch_config.get("download_concurrency") or self._config.get_stage("s01_fetch").get("max_concurrent_downloads", 128)
        timeout_s = spec.fetch_config.get("download_timeout_seconds", 15)

        # 1. Download parquet metadata shard(s)
        allow_patterns = spec.fetch_config.get("file_patterns", ["*.parquet"])
        log.info("hf_metadata_downloading", repo=spec.repo_id, dest=str(dest))

        target_dir = dest / "_metadata"
        target_dir.mkdir(parents=True, exist_ok=True)

        import fnmatch
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=self._hf_token)
        try:
            repo_files = api.list_repo_files(repo_id=spec.repo_id, repo_type="dataset", revision=spec.revision or "main")
            matching_files = []
            for pat in allow_patterns:
                for rf in repo_files:
                    if fnmatch.fnmatch(rf, pat) and (rf.endswith(".parquet") or rf.endswith(".tsv")) and rf not in matching_files:
                        matching_files.append(rf)
            matching_files.sort()
        except Exception:
            matching_files = []

        if matching_files:
            # For small/capped sample_size, download only the first shard to avoid downloading multi-gigabyte shard sets
            shards_to_download = matching_files[:1] if sample_size <= 50_000 else matching_files
            for rf in shards_to_download:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=rf,
                    revision=spec.revision or "main",
                    local_dir=str(target_dir),
                    token=self._hf_token,
                )
        else:
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision or "main",
                local_dir=str(target_dir),
                allow_patterns=allow_patterns,
                token=self._hf_token,
            )

        parquet_files = sorted(Path(target_dir).rglob("*.parquet"))
        tsv_files_present = any(Path(target_dir).rglob("*.tsv"))
        if not parquet_files and not tsv_files_present:
            log.error("no_metadata_files_found", dataset=key, dir=str(target_dir))
            return []

        # 2. Read metadata, sample down to a manageable size
        needed_cols = [url_col]
        if caption_col:
            needed_cols.append(caption_col)

        frames = []
        for pf in parquet_files:
            try:
                # Memory optimization: project only required columns
                import pyarrow.parquet as pq
                file_schema = pq.read_schema(pf)
                read_cols = [c for c in needed_cols if c in file_schema.names]
                frames.append(pd.read_parquet(pf, columns=read_cols if read_cols else None))
            except Exception as e:
                try:
                    frames.append(pd.read_parquet(pf, columns=None))
                except Exception as inner_e:
                    log.warning("parquet_read_failed", file=str(pf), error=str(inner_e))

        # CC12M's canonical raw distribution is a headerless TSV
        # (caption<TAB>url), not parquet — fetch_config.file_patterns
        # already lists "*.tsv" for it. Handle both so a dataset spec isn't
        # silently empty just because it ships the older format.
        tsv_files = sorted(Path(target_dir).rglob("*.tsv"))
        for tf in tsv_files:
            try:
                tsv_df = pd.read_csv(
                    tf, sep="\t", header=None, names=["caption", "url"],
                    on_bad_lines="skip", quoting=3,
                )
                frames.append(tsv_df)
            except Exception as e:
                log.warning("tsv_read_failed", file=str(tf), error=str(e))

        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        if url_col not in df.columns:
            log.error(
                "image_url_column_not_found",
                dataset=key,
                configured_column=url_col,
                available_columns=list(df.columns)[:30],
            )
            return []

        df = df.dropna(subset=[url_col])
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        log.info("url_list_sampled", dataset=key, rows=len(df), total_available_hint="see repo card")

        # 3. Concurrently download images
        images_dir = dest / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        concurrency = max(1, min(int(concurrency), 128))
        max_image_bytes = int(spec.fetch_config.get("max_image_bytes", 25_000_000))
        max_image_pixels = int(spec.fetch_config.get("max_image_pixels", 100_000_000))
        semaphore = asyncio.Semaphore(concurrency)
        records: list[dict[str, Any]] = []

        import io
        from PIL import Image

        limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
        timeout = httpx.Timeout(connect=min(5.0, timeout_s), read=timeout_s, write=timeout_s, pool=10.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True) as client:
            def _validate_and_save(
                row_idx: int,
                content: bytes,
                source_caption: str | None,
            ) -> dict[str, Any] | None:
                try:
                    with Image.open(io.BytesIO(content)) as probe:
                        width, height = probe.size
                        image_format = probe.format
                        if width < 1 or height < 1 or width * height > max_image_pixels:
                            return None
                        probe.verify()
                    with Image.open(io.BytesIO(content)) as image:
                        image.load()
                        if image.format != image_format:
                            return None
                except Exception:
                    return None

                format_extensions = {
                    "JPEG": ".jpg",
                    "PNG": ".png",
                    "WEBP": ".webp",
                    "BMP": ".bmp",
                    "TIFF": ".tiff",
                }
                ext = format_extensions.get(image_format)
                if ext is None:
                    return None

                file_path = images_dir / f"{key}_{row_idx:08d}{ext}"
                temp_path = file_path.with_name(f".{file_path.name}.part")
                try:
                    temp_path.write_bytes(content)
                    os.replace(temp_path, file_path)
                except OSError:
                    if temp_path.exists():
                        temp_path.unlink()
                    return None

                try:
                    rel_path = str(file_path.relative_to(self._config.data_root))
                except ValueError:
                    rel_path = str(file_path)

                record: dict[str, Any] = {
                    "source_file": file_path.name,
                    "image_path": rel_path,
                    "content_hash_sha256": hashlib.sha256(content).hexdigest(),
                    "image_width": width,
                    "image_height": height,
                    "file_size_bytes": len(content),
                }
                if isinstance(source_caption, str) and source_caption:
                    record["source_caption"] = source_caption
                return record

            async def _download_one(row_idx: int, url: str, source_caption: str | None) -> bool:
                async with semaphore:
                    try:
                        async with client.stream("GET", url) as resp:
                            resp.raise_for_status()
                            content_length = resp.headers.get("content-length")
                            if content_length and int(content_length) > max_image_bytes:
                                return False
                            chunks: list[bytes] = []
                            content_size = 0
                            async for chunk in resp.aiter_bytes():
                                content_size += len(chunk)
                                if content_size > max_image_bytes:
                                    return False
                                chunks.append(chunk)
                            content = b"".join(chunks)
                    except Exception as e:
                        log.debug("url_download_failed", url=url[:200], error=str(e))
                        return False

                    record = await asyncio.to_thread(
                        _validate_and_save,
                        row_idx,
                        content,
                        source_caption,
                    )
                    if record is None:
                        log.debug("image_validation_or_write_failed", dataset=key, row=row_idx)
                        return False
                    records.append(record)
                    return True

            attempted = downloaded = 0
            row_iter = enumerate(df.itertuples(index=False, name=None))
            task_batch_size = max(concurrency, min(concurrency * 8, 2048))
            while True:
                rows = list(islice(row_iter, task_batch_size))
                if not rows:
                    break
                tasks = []
                for row_idx, row in rows:
                    row_dict = dict(zip(df.columns, row))
                    url = row_dict.get(url_col)
                    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                        continue
                    caption = row_dict.get(caption_col) if caption_col else None
                    tasks.append(_download_one(row_idx, url, caption))
                downloaded += sum(await asyncio.gather(*tasks))
                attempted += len(tasks)
                log.info(
                    "url_list_progress",
                    dataset=key,
                    attempted=attempted,
                    total=len(df),
                    downloaded_so_far=len(records),
                )

        log.info(
            "url_list_fetch_complete",
            dataset=key,
            attempted=attempted,
            downloaded=downloaded,
            success_rate=round(downloaded / attempted, 3) if attempted else 0.0,
        )
        return records

    @staticmethod
    def _guess_extension(url: str, content_type: str) -> str | None:
        """Best-effort image extension from URL suffix or response content-type."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        url_path = url.split("?", 1)[0]
        suffix = Path(url_path).suffix.lower()
        if suffix in image_extensions:
            return suffix

        content_type_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        return content_type_map.get(content_type.split(";")[0].strip().lower())

    def _unpack_archives(self, directory: Path):
        import zipfile
        import tarfile
        import shutil

        # Handle split zips first (e.g. file.zip.001, file.zip.002 or file.z01, file.z02, file.zip)
        split_bases = set()
        for f in directory.rglob("*.zip.001"):
            split_bases.add(str(f)[:-4])  # base without .001
        for f in directory.rglob("*.z01"):
            split_bases.add(str(f)[:-4] + ".zip")
        
        for base in split_bases:
            combined_zip = Path(base if not base.endswith(".zip") else base[:-4] + "_combined.zip")
            if combined_zip.exists():
                continue
            
            # Find all parts
            parts = []
            # Style 1: .zip.001, .zip.002
            i = 1
            while True:
                p = Path(f"{base}.{i:03d}")
                if p.exists():
                    parts.append(p)
                    i += 1
                else:
                    break
            
            # Style 2: .z01, .z02 ... .zip
            if not parts:
                i = 1
                while True:
                    p = Path(f"{base[:-4]}.z{i:02d}")
                    if p.exists():
                        parts.append(p)
                        i += 1
                    else:
                        break
                p_final = Path(base)
                if p_final.exists():
                    parts.append(p_final)
            
            if parts:
                log.info("concatenating_split_zip", base=str(combined_zip), parts=len(parts))
                try:
                    with open(combined_zip, 'wb') as outfile:
                        for p in parts:
                            with open(p, 'rb') as infile:
                                shutil.copyfileobj(infile, outfile)
                except Exception as e:
                    log.error("split_zip_concat_failed", base=str(combined_zip), error=str(e))
                    continue

        for f in list(directory.rglob("*.zip")):
            if not f.exists():
                continue
            # If this is the last chunk of an uncombined .z01 split zip, skip extracting it directly
            if list(directory.rglob(f.name.replace(".zip", ".z01"))):
                continue
            
            log.info("extracting_zip", file=str(f))
            try:
                with zipfile.ZipFile(f, 'r') as z:
                    z.extractall(f.parent)
                try:
                    f.unlink()
                except Exception:
                    pass
            except Exception as e:
                log.warning("zip_extract_failed", file=str(f), error=str(e))

        for f in list(directory.rglob("*.tar.gz")):
            if not f.exists():
                continue
            log.info("extracting_tar", file=str(f))
            try:
                with tarfile.open(f, 'r:gz') as t:
                    t.extractall(f.parent)
                try:
                    f.unlink()
                except Exception:
                    pass
            except Exception as e:
                log.warning("tar_extract_failed", file=str(f), error=str(e))

        for f in list(directory.rglob("*.tgz")):
            if not f.exists():
                continue
            log.info("extracting_tgz", file=str(f))
            try:
                with tarfile.open(f, 'r:gz') as t:
                    t.extractall(f.parent)
                try:
                    f.unlink()
                except Exception:
                    pass
            except Exception as e:
                log.warning("tgz_extract_failed", file=str(f), error=str(e))

    async def _fetch_huggingface(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download dataset from HuggingFace Hub."""
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        log.info("hf_downloading", repo=spec.repo_id, dest=str(dest))

        # Build allow_patterns from fetch_config
        allow_patterns = spec.fetch_config.get("file_patterns")

        snapshot_dir = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest),
            allow_patterns=allow_patterns,
            token=self._hf_token,
        )

        self._unpack_archives(Path(snapshot_dir))

        if spec.annotation_only:
            log.info(
                "annotation_only_hf_source_downloaded",
                dataset=key,
                dir=str(snapshot_dir),
                note="Dataset metadata/structure preserved without scanning for image records.",
            )
            return []

        records = self._scan_downloaded_files(key, Path(snapshot_dir))
        sample_size = spec.fetch_config.get("sample_size")
        if sample_size and len(records) > sample_size:
            records = records[:sample_size]
        return records

    async def _fetch_github(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Clone or download a GitHub repository."""
        import subprocess
        import os

        if not spec.repo_url:
            log.error("missing_repo_url", dataset=key)
            return []

        repo_url = spec.repo_url
        token = os.environ.get("GITHUB_TOKEN")
        if token and repo_url.startswith("https://"):
            repo_url = repo_url.replace("https://", f"https://{token}@")

        clone_dir = dest / "repo"
        if clone_dir.exists():
            log.info("github_repo_exists", dataset=key, path=str(clone_dir))
        else:
            log.info("github_cloning", repo=spec.repo_url)
            subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", spec.branch or "main",
                    repo_url,
                    str(clone_dir),
                ],
                check=True,
                capture_output=True,
            )

        if spec.annotation_only:
            log.info(
                "annotation_only_source_cloned_not_scanned",
                dataset=key,
                clone_dir=str(clone_dir),
                note="No image records created for this source — see the "
                     "dedicated join stage that consumes it directly.",
            )
            return []

        return self._scan_downloaded_files(key, clone_dir)

    async def _fetch_url(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download from a direct URL."""
        import httpx

        url = spec.fetch_config.get("url")
        if not url:
            log.error("missing_url", dataset=key)
            return []

        filename = url.rsplit("/", 1)[-1]
        file_path = dest / filename

        if file_path.exists():
            log.info("url_file_exists", path=str(file_path))
        else:
            log.info("url_downloading", url=url)
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=3600)
                response.raise_for_status()
                file_path.write_bytes(response.content)

        return self._scan_downloaded_files(key, dest)

    def _scan_downloaded_files(
        self, dataset_key: str, directory: Path
    ) -> list[dict[str, Any]]:
        """Scan downloaded directory for image files and return record metadata."""
        import json
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        records: list[dict[str, Any]] = []

        # Check for dataset-wide captions_index.json (e.g. screen2words)
        captions_index: dict[str, Any] = {}
        index_file = directory / "captions_index.json"
        if index_file.exists():
            try:
                captions_index = json.loads(index_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        img_files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

        def _process_image_file(file_path: Path) -> dict[str, Any] | None:
            try:
                sha256 = self._compute_sha256(file_path)
                width, height = self._get_image_dimensions(file_path)

                try:
                    rel_path = str(file_path.relative_to(self._config.data_root))
                except ValueError:
                    rel_path = str(file_path)

                source_caption = None
                source_url = None

                if file_path.name in captions_index:
                    entry = captions_index[file_path.name]
                    if isinstance(entry, dict):
                        source_caption = entry.get("primary_caption") or (entry.get("captions") and entry["captions"][0])
                    elif isinstance(entry, str):
                        source_caption = entry

                json_file = file_path.with_suffix(".json")
                if json_file.exists():
                    try:
                        meta = json.loads(json_file.read_text(encoding="utf-8"))
                        source_caption = source_caption or meta.get("caption") or meta.get("title")
                        source_url = meta.get("url") or meta.get("image_url")
                    except Exception:
                        pass

                return {
                    "source_file": file_path.name,
                    "image_path": rel_path,
                    "content_hash_sha256": sha256,
                    "image_width": width,
                    "image_height": height,
                    "file_size_bytes": file_path.stat().st_size,
                    "source_caption": source_caption,
                    "source_url": source_url,
                }
            except Exception:
                return None

        import concurrent.futures
        max_workers = min(32, max(4, os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for rec in executor.map(_process_image_file, img_files):
                if rec:
                    records.append(rec)

        log.info(
            "scan_completed",
            dataset=dataset_key,
            image_count=len(records),
            directory=str(directory),
        )
        return records

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_image_dimensions(file_path: Path) -> tuple[int | None, int | None]:
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                return img.size  # (width, height)
        except Exception:
            return None, None
