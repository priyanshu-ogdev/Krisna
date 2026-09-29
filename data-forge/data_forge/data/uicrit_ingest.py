"""UICrit annotation parsing and RICO-record join.

UICrit (google-research-datasets/uicrit) is NOT a standalone image
dataset — its screenshots are RICO's screenshots. Its actual value is
~983 human-authored critiques and numeric ratings, keyed by RICO screen
ID. The generic image-glob fetch path (`_scan_downloaded_files`) can
never surface this data because it only recognizes image file extensions
and has no concept of a join — see `fetcher.py::_fetch_github`'s
`annotation_only` handling for the other half of this fix.

UNVERIFIED, explicitly: the exact filename and column names UICrit's
repo uses for (a) the annotation file itself and (b) the RICO screen-ID
join key are not independently confirmed against the live repo as of
this revision — GitHub repo contents can also change. Rather than
silently guessing and producing zero matches with no explanation (the
exact failure mode this module exists to fix), `parse_uicrit_annotations`
tries a documented list of candidate filenames/columns and logs plainly
which one worked, or logs every candidate it tried and why each failed if
none did. Confirm the real names against the live repo before a
production run and, if they differ, add the actual names to the front of
the candidate lists below — don't silently rely on the fallback matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("data.uicrit_ingest")

# Candidate annotation filenames, most-likely-first. UICrit's repo is
# small (~983 rows) so a single CSV or JSON file is the expected shape;
# glob patterns catch reasonable variations without hardcoding one exact
# path that breaks the moment the repo's layout shifts.
_ANNOTATION_FILE_GLOBS = [
    "*ratings*.csv", "*ratings*.json",
    "*critique*.csv", "*critique*.json",
    "*annotations*.csv", "*annotations*.json",
    "data/*.csv", "data/*.json",
    "*.csv", "*.json",  # last resort: any top-level tabular file
]

# Candidate column names for the RICO screen-ID join key, most-likely-first.
_ID_COLUMN_CANDIDATES = [
    "rico_id", "rico_screen_id", "screen_id", "image_id", "id", "filename",
]

# Candidate column names for free-text critique and numeric rating fields.
_CRITIQUE_TEXT_CANDIDATES = [
    "critique", "comment", "feedback", "review", "annotation", "text",
]
_RATING_CANDIDATES = [
    "rating", "score", "quality_rating", "aesthetics_rating", "overall_rating",
]


def find_annotation_file(repo_dir: Path) -> Path | None:
    """Locate UICrit's annotation file inside the cloned repo."""
    for pattern in _ANNOTATION_FILE_GLOBS:
        matches = sorted(repo_dir.rglob(pattern))
        # Skip anything that's obviously not annotation data (e.g. a
        # requirements.csv or a LICENSE-adjacent file that happens to
        # match "*.csv") by requiring a minimum file size — a real
        # ~983-row annotation file won't be a few bytes.
        matches = [m for m in matches if m.stat().st_size > 1024]
        if matches:
            log.info("uicrit_annotation_file_found", pattern=pattern, path=str(matches[0]))
            return matches[0]
    log.error(
        "uicrit_annotation_file_not_found",
        repo_dir=str(repo_dir),
        tried_patterns=_ANNOTATION_FILE_GLOBS,
        note="No file matched any candidate pattern above 1KB. Inspect "
             "the cloned repo manually and add its real path/pattern to "
             "_ANNOTATION_FILE_GLOBS.",
    )
    return None


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def parse_uicrit_annotations(repo_dir: Path) -> list[dict[str, Any]]:
    """Parse UICrit's annotation file into a list of normalized dicts.

    Returns:
        List of {rico_join_key, critique_text, rating, raw_fields} dicts.
        Empty list if the file couldn't be found or parsed — callers
        should treat that as a hard stop, not silently proceed with zero
        annotations as if that were expected.
    """
    import pandas as pd

    ann_path = find_annotation_file(repo_dir)
    if ann_path is None:
        return []

    try:
        if ann_path.suffix == ".csv":
            df = pd.read_csv(ann_path)
        else:
            df = pd.read_json(ann_path)
    except Exception as e:
        log.error("uicrit_annotation_parse_failed", path=str(ann_path), error=str(e))
        return []

    columns = list(df.columns)
    id_col = _pick_column(columns, _ID_COLUMN_CANDIDATES)
    text_col = _pick_column(columns, _CRITIQUE_TEXT_CANDIDATES)
    rating_col = _pick_column(columns, _RATING_CANDIDATES)

    if id_col is None:
        log.error(
            "uicrit_id_column_not_found",
            available_columns=columns,
            tried=_ID_COLUMN_CANDIDATES,
            note="Cannot join without a screen-ID column. Inspect "
                 f"{ann_path} manually and add the real column name to "
                 "_ID_COLUMN_CANDIDATES.",
        )
        return []

    log.info(
        "uicrit_columns_resolved",
        id_col=id_col,
        text_col=text_col,
        rating_col=rating_col,
        total_rows=len(df),
        all_columns=columns,
    )

    out = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        rico_key = str(raw.get(id_col, "")).strip()
        if not rico_key:
            continue
        out.append({
            "rico_join_key": rico_key,
            "critique_text": str(raw.get(text_col, "")) if text_col else "",
            "rating": raw.get(rating_col) if rating_col else None,
            "raw_fields": {k: v for k, v in raw.items() if k not in (id_col,)},
        })
    return out


def to_critique_output_dict(annotation: dict[str, Any]) -> dict[str, Any]:
    """Best-effort mapping of a UICrit row onto the pipeline's CritiqueOutput
    shape (see inference/structured_output.py), so human-sourced and
    Gemma-4-sourced critiques share one schema downstream.

    UICrit's actual rubric almost certainly doesn't map cleanly 1:1 onto
    the four dimensions Gemma 4 is prompted for (visual_hierarchy,
    readability, layout_consistency, brand_alignment) — this is a
    best-effort normalization, not a claim of equivalence. The full raw
    row is preserved under `raw_fields` specifically so nothing is lost
    if the mapping below turns out to be wrong once the real schema is
    confirmed.
    """
    rating = annotation.get("rating")
    try:
        normalized_score = float(rating) / 5.0 if rating is not None else 0.5
        normalized_score = max(0.0, min(1.0, normalized_score))
    except (TypeError, ValueError):
        normalized_score = 0.5

    note = (annotation.get("critique_text") or "")[:400]

    return {
        "critique_source": "uicrit_human",
        "overall_score": normalized_score,
        "visual_hierarchy_score": normalized_score,
        "visual_hierarchy_note": note,
        "readability_score": normalized_score,
        "readability_note": note,
        "layout_consistency_score": normalized_score,
        "layout_consistency_note": note,
        "brand_alignment_score": normalized_score,
        "brand_alignment_note": note,
        "suggested_edits": [],
        "raw_fields": annotation.get("raw_fields", {}),
    }
