import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.continuity_manager import ContinuityManager
from storybook.state import NovelSystemState

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="creation",
        project=Mock(
            content={
                "timeline": {},
                "characters": {},
                "plot_threads": []
            }
        ),
        current_input={"task": "timeline_check"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_timeline_processing(mock_state):
    """Test timeline processing functionality."""
    manager = ContinuityManager()
    result = await manager.process(mock_state, {})
    
    assert "messages" in result
    assert "continuity_updates" in result
    assert "timeline" in result["continuity_updates"]

@pytest.mark.asyncio
async def test_character_tracking(mock_state):
    """Test character tracking functionality."""
    mock_state.current_input["task"] = "character_tracking"
    manager = ContinuityManager()
    result = await manager.process(mock_state, {})
    
    assert "messages" in result
    assert "continuity_updates" in result
    assert "character_tracking" in result["continuity_updates"]

@pytest.mark.asyncio
async def test_plot_consistency(mock_state):
    """Test plot consistency checking."""
    mock_state.current_input["task"] = "plot_consistency"
    manager = ContinuityManager()
    result = await manager.process(mock_state, {})
    
    assert "messages" in result
    assert "continuity_updates" in result
    assert "plot_consistency" in result["continuity_updates"]
