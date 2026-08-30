"""Domain-aware completeness predicates for encoded records.

Stage 8 (Encoding) produces a different artifact set depending on a
record's domain: `vq_tokens` (sketch-tier training data) is only computed
for `ui_first` records — see s08_encoding.py's domain gate. A
`general_design` record correctly having two artifacts instead of three is
NOT an incompleteness — it's the intended, resource-efficient behavior.
Any code checking "did this record encode successfully" needs to know
that distinction rather than applying one blanket rule, which would
incorrectly flag every properly-processed general_design record as
broken.

UPDATED (data-pipeline upgrade): `qwen_image_latent` was dropped from
`_REQUIRED_ALWAYS` — Qwen-Image-Edit-2511 ships frozen now, and
s08_encoding.py no longer produces that artifact for any record (see its
Branch 2 removal comment). Leaving it in this set would have made
`is_encoding_complete()` return False for every record in the corpus,
silently starving s09_heldout/s10_audit/s12_model_data_export of any
input at all — exactly the class of drift this module's docstring warns
about, just triggered by a removal instead of an addition this time.
"""

from __future__ import annotations

from data_forge.manifest import ManifestRecord

# Every record, regardless of domain, needs these two.
_REQUIRED_ALWAYS = frozenset({"z_image_latent", "control_map"})
# Only ui_first records are expected to have this one.
_REQUIRED_UI_FIRST_ONLY = frozenset({"vq_tokens"})


def required_artifacts_for(domain: str | None) -> frozenset[str]:
    """The complete artifact set a record of this domain should have."""
    if domain == "ui_first":
        return _REQUIRED_ALWAYS | _REQUIRED_UI_FIRST_ONLY
    return _REQUIRED_ALWAYS


def is_encoding_complete(rec: ManifestRecord) -> bool:
    """Whether a record has every artifact its domain requires.

    Used wherever downstream code (heldout carve, audit sampling, the
    model-data export) needs to trust that a record's encoded artifacts
    are actually all present, not just that status == "encoded" (which
    Stage 8 sets as soon as ANY artifact succeeds, per its own "don't
    silently produce a record with zero artifacts" rule — that's a
    weaker guarantee than "produced everything this domain needs").
    """
    if not rec.encoding_paths:
        return False
    required = required_artifacts_for(rec.domain)
    return required.issubset(rec.encoding_paths.keys())


def missing_artifacts(rec: ManifestRecord) -> frozenset[str]:
    """Which required artifacts (if any) this record is missing."""
    if not rec.encoding_paths:
        return required_artifacts_for(rec.domain)
    required = required_artifacts_for(rec.domain)
    return required - rec.encoding_paths.keys()
