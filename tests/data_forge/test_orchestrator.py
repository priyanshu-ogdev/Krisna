import pytest
from data_forge.config import PipelineConfig, PathsConfig, StageConfig
from data_forge.manifest import Manifest
from data_forge.orchestrator import Orchestrator

@pytest.fixture
def config(tmp_path):
    c = PipelineConfig()
    c.data_root = tmp_path
    c.paths = PathsConfig(checkpoints="checkpoints")
    c.stages = {
        "s05_recaption": StageConfig(enabled=True),
        "s10_audit": StageConfig(enabled=True)
    }
    return c

def test_orchestrator_checkpointing(config, tmp_path):
    manifest = Manifest(tmp_path / "manifest.db")
    orch = Orchestrator(config, manifest)
    
    # Assert missing checkpoint
    assert orch._is_stage_complete("s05_recaption", "chunk_0001") is False
    
    # Mark stage as complete
    orch._mark_stage_complete("s05_recaption", "chunk_0001")
    
    # Verify checkpoint detection
    assert orch._is_stage_complete("s05_recaption", "chunk_0001") is True
