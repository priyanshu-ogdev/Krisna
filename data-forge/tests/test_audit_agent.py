import pytest

from data_forge.agents.audit_agent import AuditAgent
from data_forge.inference.structured_output import AuditOutput
from data_forge.manifest import ManifestRecord

def test_ensemble_disagreement_caption():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe", caption_output={"confidence": 0.9})
    
    # Audit says no match, but stage 5 was highly confident
    audit_out = AuditOutput(
        caption_matches_image=False, 
        structure_matches_image=True, 
        safety_issues=False, 
        overall_pass=False, 
        reasoning="Bad caption"
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_disagreement_structure():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe", structure_output={"elements": []})
    
    audit_out = AuditOutput(
        caption_matches_image=True, 
        structure_matches_image=False, 
        safety_issues=False, 
        overall_pass=False, 
        reasoning="Bad structure"
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_disagreement_safety():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe")
    
    # Audit says safety issues but stage 4 said "safe"
    audit_out = AuditOutput(
        caption_matches_image=True, 
        structure_matches_image=True, 
        safety_issues=True, 
        overall_pass=False, 
        reasoning="NSFW found"
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is True

def test_ensemble_agreement():
    agent = AuditAgent(config=None) # type: ignore
    rec = ManifestRecord(id="test1", source_dataset="rico", safety_tier="safe", caption_output={"confidence": 0.9})
    
    audit_out = AuditOutput(
        caption_matches_image=True, 
        structure_matches_image=True, 
        safety_issues=False, 
        overall_pass=True, 
        reasoning="Looks good"
    )
    assert agent.check_ensemble_disagreement(rec, audit_out) is False
