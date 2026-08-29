"""Dataset download manager — HuggingFace Hub, GitHub, and direct URL.

Handles dataset fetching with resume support, checksum verification,
and progress tracking via manifest updates.
"""

from __future__ import annotations

import hashlib
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

        # New: format-teacher reference datasets (Glaive Function Calling
        # v2, Salesforce xLAM) for the planner conversational-synthesis
        # stage. These are pure text (conversation/tool-call JSON), not
        # images — downloaded once as raw JSONL/parquet reference material
        # for s01_6_planner_synthesis.py to read directly, not inserted into
        # the image-record manifest at all (same "return []" pattern as
        # annotation_only sources, for the same reason: this isn't image
        # data and doesn't belong in a table whose schema assumes it is).
        if spec.fetch_config.get("download_mode") == "text_reference":
            return await self._fetch_text_reference(key, spec, dataset_dir)

        # New: paired (source, instruction, target) edit datasets
        # (MagicBrush, InstructPix2Pix) — writes directly into
        # processed/edit_pairs/, not the image-record manifest, since a
        # pair isn't a single-image record. See s07_5_edit_pairs.py, which
        # combines this general-domain data with synthetically-constructed
        # UI-domain pairs.
        if spec.fetch_config.get("download_mode") == "triple_dataset":
            return await self._fetch_huggingface_triples(key, spec, dataset_dir)

        if spec.source_type == "huggingface":
            return await self._fetch_huggingface(key, spec, dataset_dir)
        elif spec.source_type == "github":
            return await self._fetch_github(key, spec, dataset_dir)
        elif spec.source_type == "url":
            return await self._fetch_url(key, spec, dataset_dir)
        else:
            log.error("unknown_source_type", dataset=key, source_type=spec.source_type)
            return []

    async def _fetch_huggingface_triples(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Fetch a (source_image, instruction, target_image) triple dataset
        and write pairs directly into processed/edit_pairs/.

        UNVERIFIED column names, same auto-detection discipline as
        _fetch_huggingface_caption_join: looks for two embedded-image-like
        columns and one text column, logs what it found, and logs loudly
        (not silently) if it can't confidently identify all three rather
        than guessing and producing garbage pairs.
        """
        import io

        import pandas as pd
        from huggingface_hub import snapshot_download
        from PIL import Image

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        meta_dir = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest / "_metadata"),
            allow_patterns=spec.fetch_config.get("file_patterns", ["*.parquet"]),
            token=self._hf_token,
        )
        parquet_files = sorted(Path(meta_dir).rglob("*.parquet"))
        if not parquet_files:
            log.error("triple_dataset_no_parquet", dataset=key, dir=str(meta_dir))
            return []

        frames = []
        for pf in parquet_files:
            try:
                frames.append(pd.read_parquet(pf))
            except Exception as e:
                log.warning("triple_dataset_parquet_read_failed", file=str(pf), error=str(e))
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

        text_col = next(
            (c for c in columns if c.lower() in
             ("instruction", "edit_instruction", "prompt", "edit_prompt", "text")),
            None,
        )

        if len(image_cols) < 2 or text_col is None:
            log.error(
                "triple_dataset_undetectable",
                dataset=key,
                available_columns=columns,
                detected_image_cols=image_cols,
                detected_text_col=text_col,
                note="Need exactly 2 image-like columns (source, target) and "
                     "1 instruction text column — inspect the real schema and "
                     "override via fetch_config if auto-detection guessed wrong.",
            )
            return []

        source_col, target_col = image_cols[0], image_cols[1]
        log.info(
            "triple_dataset_columns_resolved",
            dataset=key, source_col=source_col, target_col=target_col, text_col=text_col,
        )

        sample_size = spec.fetch_config.get("sample_size")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        out_dir = self._config.resolved_paths["edit_pairs"] / "external" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = dict(zip(df.columns, row))
            try:
                src_raw = row_dict.get(source_col)
                tgt_raw = row_dict.get(target_col)
                src_blob = src_raw.get("bytes") if isinstance(src_raw, dict) else src_raw
                tgt_blob = tgt_raw.get("bytes") if isinstance(tgt_raw, dict) else tgt_raw
                if not src_blob or not tgt_blob:
                    continue
                src_img = Image.open(io.BytesIO(src_blob)); src_img.load()
                tgt_img = Image.open(io.BytesIO(tgt_blob)); tgt_img.load()
            except Exception:
                continue

            pair_id = f"{key}_{idx:07d}"
            src_img.save(out_dir / f"{pair_id}_source.png", "PNG")
            tgt_img.save(out_dir / f"{pair_id}_target.png", "PNG")
            (out_dir / f"{pair_id}.json").write_text(
                json.dumps({
                    "pair_id": pair_id,
                    "instruction": str(row_dict.get(text_col, "")),
                    "source": f"{pair_id}_source.png",
                    "target": f"{pair_id}_target.png",
                    "origin": key,
                }),
                encoding="utf-8",
            )
            written += 1

        log.info("triple_dataset_pairs_written", dataset=key, count=written)
        return []  # Not manifest records — written directly to edit_pairs/

    async def _fetch_text_reference(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download a pure-text reference dataset (no images) as-is.

        Used for format-teacher datasets like Glaive Function Calling v2
        and Salesforce xLAM — real, open, well-formed tool-call/JSON
        examples that teach output *shape*, general-domain (nothing
        public teaches "UI design conversation" specifically — see
        s01_6_planner_synthesis.py's docstring). Files are left in their
        native format under raw/{key}/ for that stage to read directly;
        no manifest records are created since these aren't images.
        """
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        log.info("text_reference_downloading", repo=spec.repo_id, dest=str(dest))
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest),
            allow_patterns=spec.fetch_config.get("file_patterns", ["*.parquet", "*.json", "*.jsonl"]),
            token=self._hf_token,
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

        meta_dir = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest / "_metadata"),
            allow_patterns=spec.fetch_config.get("file_patterns", ["*.parquet"]),
            token=self._hf_token,
        )
        parquet_files = sorted(Path(meta_dir).rglob("*.parquet"))
        if not parquet_files:
            log.error("caption_join_no_parquet_found", dataset=key, dir=str(meta_dir))
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

        if image_col is not None:
            log.info("caption_join_path_a_embedded_images", dataset=key, image_col=image_col, caption_col=caption_col)
            return self._decode_embedded_images(key, dest, df, image_col, caption_col)

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
        self, key: str, dest: Path, df: Any, image_col: str, caption_col: str | None
    ) -> list[dict[str, Any]]:
        """Decode a parquet column of embedded image bytes to files on disk."""
        import io

        from PIL import Image

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
        concurrency = spec.fetch_config.get("download_concurrency", 32)
        timeout_s = spec.fetch_config.get("download_timeout_seconds", 15)

        # 1. Download parquet metadata shard(s)
        allow_patterns = spec.fetch_config.get("file_patterns", ["*.parquet"])
        log.info("hf_metadata_downloading", repo=spec.repo_id, dest=str(dest))
        meta_dir = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest / "_metadata"),
            allow_patterns=allow_patterns,
            token=self._hf_token,
        )

        parquet_files = sorted(Path(meta_dir).rglob("*.parquet"))
        tsv_files_present = any(Path(meta_dir).rglob("*.tsv"))
        if not parquet_files and not tsv_files_present:
            log.error("no_metadata_files_found", dataset=key, dir=str(meta_dir))
            return []

        # 2. Read metadata, sample down to a manageable size
        frames = []
        for pf in parquet_files:
            try:
                frames.append(pd.read_parquet(pf, columns=None))
            except Exception as e:
                log.warning("parquet_read_failed", file=str(pf), error=str(e))

        # CC12M's canonical raw distribution is a headerless TSV
        # (caption<TAB>url), not parquet — fetch_config.file_patterns
        # already lists "*.tsv" for it. Handle both so a dataset spec isn't
        # silently empty just because it ships the older format.
        tsv_files = sorted(Path(meta_dir).rglob("*.tsv"))
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
        semaphore = asyncio.Semaphore(concurrency)
        records: list[dict[str, Any]] = []
        records_lock = asyncio.Lock()

        async def _download_one(row_idx: int, url: str, source_caption: str | None) -> None:
            async with semaphore:
                try:
                    async with httpx.AsyncClient(follow_redirects=True) as client:
                        resp = await client.get(url, timeout=timeout_s)
                        resp.raise_for_status()
                        content = resp.content
                except Exception as e:
                    log.debug("url_download_failed", url=url[:200], error=str(e))
                    return

                # Derive a filename from the row index (URLs often lack a
                # usable extension or collide in basename across rows).
                ext = self._guess_extension(url, resp.headers.get("content-type", ""))
                if ext is None:
                    return  # not an image / undecodable content-type
                file_path = images_dir / f"{key}_{row_idx:08d}{ext}"

                try:
                    file_path.write_bytes(content)
                    from PIL import Image
                    with Image.open(file_path) as img:
                        width, height = img.size
                except Exception as e:
                    log.debug("image_decode_failed", url=url[:200], error=str(e))
                    file_path.unlink(missing_ok=True)
                    return

                sha256 = self._compute_sha256(file_path)
                try:
                    rel_path = str(file_path.relative_to(self._config.data_root))
                except ValueError:
                    rel_path = str(file_path)

                record: dict[str, Any] = {
                    "source_file": file_path.name,
                    "image_path": rel_path,
                    "content_hash_sha256": sha256,
                    "image_width": width,
                    "image_height": height,
                    "file_size_bytes": file_path.stat().st_size,
                }
                if source_caption:
                    record["source_caption"] = source_caption

                async with records_lock:
                    records.append(record)

        tasks = []
        for row_idx, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
            url = row_dict.get(url_col)
            if not url:
                continue
            caption = row_dict.get(caption_col) if caption_col else None
            tasks.append(_download_one(row_idx, url, caption))

        for i in range(0, len(tasks), 5000):
            batch = tasks[i : i + 5000]
            await asyncio.gather(*batch)
            log.info(
                "url_list_progress",
                dataset=key,
                attempted=min(i + 5000, len(tasks)),
                total=len(tasks),
                downloaded_so_far=len(records),
            )

        log.info(
            "url_list_fetch_complete",
            dataset=key,
            attempted=len(tasks),
            downloaded=len(records),
            success_rate=round(len(records) / len(tasks), 3) if tasks else 0.0,
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

        return self._scan_downloaded_files(key, Path(snapshot_dir))

    async def _fetch_github(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Clone or download a GitHub repository."""
        import subprocess

        if not spec.repo_url:
            log.error("missing_repo_url", dataset=key)
            return []

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
                    spec.repo_url,
                    str(clone_dir),
                ],
                check=True,
                capture_output=True,
            )

        # BUG FIX / COMPLETENESS GAP: this used to unconditionally call
        # _scan_downloaded_files() here regardless of what the source
        # actually contains — which hardcodes an image-extension allowlist
        # and has no awareness of fetch_config.file_patterns at all. For a
        # source like UICrit, whose entire value IS its .json/.csv
        # annotation files (not images — its screenshots are RICO's, not
        # its own), this silently cloned the repo, found zero files
        # matching the image allowlist, and returned zero records — with
        # no signal that the annotation data (983 human critiques/ratings,
        # the exact thing the PRD calls out as seeding DPO/critic
        # calibration) was ever even looked at, let alone ingested.
        # annotation_only sources skip generic image scanning entirely —
        # a dedicated stage (uicrit_ingest.py / s01_5_uicrit_join.py) reads
        # the clone directly and joins it against already-ingested image
        # records instead of pretending it's a standalone image dataset.
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
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        records: list[dict[str, Any]] = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in image_extensions:
                continue

            # Compute SHA-256
            sha256 = self._compute_sha256(file_path)

            # Get image dimensions
            width, height = self._get_image_dimensions(file_path)

            # Relative path from DATA_ROOT
            try:
                rel_path = str(file_path.relative_to(self._config.data_root))
            except ValueError:
                rel_path = str(file_path)

            records.append({
                "source_file": file_path.name,
                "image_path": rel_path,
                "content_hash_sha256": sha256,
                "image_width": width,
                "image_height": height,
                "file_size_bytes": file_path.stat().st_size,
            })

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
