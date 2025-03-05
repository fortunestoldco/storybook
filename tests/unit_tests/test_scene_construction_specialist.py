import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.scene_construction_specialist import SceneConstructionSpecialist
from storybook.state import NovelSystemState

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="creation",
        project=Mock(
            content={
                "scenes": {},
                "scene_context": {}
            }
        ),
        current_input={
            "task": {
                "type": "construct_scene",
                "scene_id": "scene_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_scene_construction():
    """Test scene construction functionality."""
    specialist = SceneConstructionSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "scene_updates" in result
    assert "scene" in result["scene_updates"]

@pytest.mark.asyncio
async def test_flow_analysis():
    """Test scene flow analysis."""
    mock_state.current_input["task"]["type"] = "flow_analysis"
    specialist = SceneConstructionSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "scene_updates" in result
    assert "flow" in result["scene_updates"]

@pytest.mark.asyncio
async def test_transition_crafting():
    """Test scene transition crafting."""
    mock_state.current_input["task"]["type"] = "transition"
    specialist = SceneConstructionSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "scene_updates" in result
    assert "transition" in result["scene_updates"]