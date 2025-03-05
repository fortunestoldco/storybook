import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.emotional_arc_designer import EmotionalArcDesigner
from storybook.state import NovelSystemState

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="creation",
        project=Mock(
            content={
                "characters": {}
            }
        ),
        current_input={
            "task": {
                "character_id": "character_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_emotional_arc_design(mock_state):
    """Test emotional arc design functionality."""
    designer = EmotionalArcDesigner()
    result = await designer.process(mock_state, {})
    
    assert "messages" in result
    assert "emotion_updates" in result
    assert "arc" in result["emotion_updates"]

@pytest.mark.asyncio
async def test_no_character_id(mock_state):
    """Test handling of missing character ID."""
    mock_state.current_input["task"]["character_id"] = ""
    designer = EmotionalArcDesigner()
    result = await designer.process(mock_state, {})
    
    assert "messages" in result
    assert "emotion_updates" in result
    assert "arc" not in result["emotion_updates"]
