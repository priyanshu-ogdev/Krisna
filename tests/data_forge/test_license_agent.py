import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from data_forge.agents.license_agent import LicenseVerificationAgent
from data_forge.inference.structured_output import LicenseOutput
from data_forge.config import PipelineConfig

@pytest.fixture
def config(tmp_path):
    c = PipelineConfig()
    c.data_root = tmp_path
    return c

@pytest.mark.asyncio
async def test_contradictory_license_text(config):
    agent = LicenseVerificationAgent(config=config, confidence_threshold=0.85)
    
    mock_engine = AsyncMock()
    # Simulate the LLM being confused by contradictory text
    mock_engine.verify_license.return_value = LicenseOutput(
        license_type="Custom/Contradictory",
        confidence=0.45,
        commercial_use_allowed=True,
        redistribution_allowed=True,
        attribution_required=True,
        research_only=True, # Conflicting with commercial
        key_restrictions=["Unclear commercial terms"],
        summary="The text states it is for commercial use but also strictly research only.",
        source_citation="Contradictory text snippet"
    )

    with patch("data_forge.agents.license_agent.fetch_page_text", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "<html>You can use this commercially. But strictly for research only.</html>"
        
        result = await agent.verify_dataset_license(
            dataset_key="test_dataset",
            license_url="http://example.com/license",
            tier1_engine=mock_engine,
        )

    assert result["verified"] is False
    assert "Low confidence" in result["reason"]
    assert "Research-only restriction" in result["reason"]
    assert mock_engine.verify_license.called
