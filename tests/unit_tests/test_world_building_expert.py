import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.world_building_expert import WorldBuildingExpert
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "world": {
                "elements": {},
                "consistency": {},
                "locations": {}
            }
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="development",
        project=mock_project,
        current_input={
            "task": {
                "type": "verify_consistency",
                "world": "fantasy"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_world_consistency(mock_state):
    """Test world consistency verification."""
    expert = WorldBuildingExpert()
    result = await expert.process(mock_state, {})
    
    assert "messages" in result
    assert "world_updates" in result
    assert "consistency" in result["world_updates"]

@pytest.mark.asyncio
async def test_location_update(mock_state):
    """Test location update functionality."""
    mock_state.current_input["task"]["type"] = "update_location"
    expert = WorldBuildingExpert()
    result = await expert.process(mock_state, {})
    
    assert "messages" in result
    assert "world_updates" in result
    assert "locations" in result["world_updates"]

@pytest.mark.asyncio
async def test_world_design(mock_state):
    """Test world design functionality."""
    mock_state.current_input["task"]["type"] = "design_world"
    expert = WorldBuildingExpert()
    result = await expert.process(mock_state, {})
    
    assert "messages" in result
    assert "world_updates" in result
    assert "elements" in result["world_updates"]
