"""Regression tests for s10_5_critic_preference.py's sampling eligibility.

Covers the bug traced during the v10 PRD completeness review: this stage
used to sample from every audited/training_pool record with no regard
for whether a record already carried real human ground truth from
s01_5_uicrit_join. Since Manifest.update_record() overwrites
critique_output wholesale rather than merging, a RICO/UICrit record
randomly selected here would have its human label silently replaced with
a Gemma-4-generated one — destroying the exact "human calibration, not
self-distillation" signal s01_5_uicrit_join exists to produce.
"""

from __future__ import annotations

from data_forge.manifest import Manifest
from data_forge.stages.s10_5_critic_preference import (
    _has_human_critique,
    _is_eligible_for_critic_sampling,
)


def _make_record(manifest: Manifest, *, status: str, critique_output=None):
    rec = manifest.create_record(
        source_dataset="rico_core", source_file="x.jpg", image_path="raw/rico_core/x.jpg"
    )
    manifest.update_record(rec.id, "test_setup", new_status=status, critique_output=critique_output)
    return manifest.get_record(rec.id)


class TestHasHumanCritique:
    def test_true_for_uicrit_human_source(self, manifest: Manifest):
        rec = _make_record(
            manifest, status="training_pool",
            critique_output={"critique_source": "uicrit_human", "overall_score": 0.8},
        )
        assert _has_human_critique(rec) is True

    def test_false_for_gemma_generated_source(self, manifest: Manifest):
        rec = _make_record(
            manifest, status="training_pool",
            critique_output={"critique_source": "gemma4_31b", "overall_score": 0.7},
        )
        assert _has_human_critique(rec) is False

    def test_false_for_no_critique_at_all(self, manifest: Manifest):
        rec = _make_record(manifest, status="training_pool", critique_output=None)
        assert _has_human_critique(rec) is False


class TestEligibilityExcludesHumanLabeledRecords:
    def test_uicrit_human_record_is_not_eligible(self, manifest: Manifest):
        """The core regression case: a RICO record that already has a
        real human critique from s01_5_uicrit_join must never be
        eligible for this stage's sampling, or it risks having that
        human label overwritten by a self-generated one."""
        rec = _make_record(
            manifest, status="training_pool",
            critique_output={"critique_source": "uicrit_human", "overall_score": 0.9},
        )
        assert _is_eligible_for_critic_sampling(rec) is False

    def test_ungrounded_record_is_eligible(self, manifest: Manifest):
        """A record with no prior critique at all should still be
        eligible — the fix must not accidentally exclude everything."""
        rec = _make_record(manifest, status="training_pool", critique_output=None)
        assert _is_eligible_for_critic_sampling(rec) is True

    def test_previously_gemma_labeled_record_remains_eligible(self, manifest: Manifest):
        """Re-running this stage on a record that already has a prior
        Gemma-4 critique (not human) is fine — only human ground truth
        is protected."""
        rec = _make_record(
            manifest, status="audited",
            critique_output={"critique_source": "gemma4_31b", "overall_score": 0.6},
        )
        assert _is_eligible_for_critic_sampling(rec) is True

    def test_wrong_status_is_not_eligible_regardless_of_critique(self, manifest: Manifest):
        """Status gating still applies independently of the human-label
        safeguard — a record that hasn't reached audited/training_pool
        yet shouldn't be sampled either."""
        rec = _make_record(manifest, status="fetched", critique_output=None)
        assert _is_eligible_for_critic_sampling(rec) is False

    def test_mixed_pool_only_excludes_the_human_labeled_ones(self, manifest: Manifest):
        """End-to-end sanity check on a small mixed pool: exactly the
        uicrit_human record is excluded, everything else survives."""
        human = _make_record(
            manifest, status="training_pool",
            critique_output={"critique_source": "uicrit_human", "overall_score": 0.85},
        )
        ungrounded = _make_record(manifest, status="training_pool", critique_output=None)
        gemma_prior = _make_record(
            manifest, status="audited",
            critique_output={"critique_source": "gemma4_31b", "overall_score": 0.5},
        )

        pool = [human, ungrounded, gemma_prior]
        eligible_ids = {r.id for r in pool if _is_eligible_for_critic_sampling(r)}

        assert human.id not in eligible_ids
        assert ungrounded.id in eligible_ids
        assert gemma_prior.id in eligible_ids
