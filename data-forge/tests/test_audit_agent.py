import pytest

from data_forge.agents.audit_agent import AuditAgent
from data_forge.inference.structured_output import AuditOutput
from data_forge.manifest import ManifestRecord

# NOTE: these fixtures were drifted against the real AuditOutput schema
# (structured_output.py) — `reasoning` -> `rationale`, `safety_issues`
# bool -> list[str], and two required fields (`confidence`, `rationale`)
# were missing entirely. Fixed to match the live schema rather than
# loosening the schema to match stale tests.

def test_ensemble_disagreement_caption():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe", caption_output={"confidence": 0.9})

    # Audit says no match, but stage 5 was highly confident
    audit_out = AuditOutput(
        caption_matches_image=False,
        structure_matches_image=True,
        safety_issues=[],
        overall_pass=False,
        confidence=0.9,
        rationale="Caption does not match image content",
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_disagreement_structure():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", structure_output={"elements": []})

    audit_out = AuditOutput(
        caption_matches_image=True,
        structure_matches_image=False,
        safety_issues=[],
        overall_pass=False,
        confidence=0.9,
        rationale="Structure does not match image content",
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_disagreement_safety():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe")

    # Audit says safety issues but stage 4 said "safe"
    audit_out = AuditOutput(
        caption_matches_image=True,
        structure_matches_image=True,
        safety_issues=["nsfw_content"],
        overall_pass=False,
        confidence=0.9,
        rationale="NSFW content found despite stage 4 clearing it",
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_agreement():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe", caption_output={"confidence": 0.9})

    audit_out = AuditOutput(
        caption_matches_image=True,
        structure_matches_image=True,
        safety_issues=[],
        overall_pass=True,
        confidence=0.95,
        rationale="Caption, structure, and safety all agree with prior stages",
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is False
