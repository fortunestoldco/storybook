import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.dialogue_crafter import DialogueCrafter
from storybook.state import NovelSystemState

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="creation",
        project=Mock(
            content={
                "dialogue": {},
                "characters": {}
            }
        ),
        current_input={
            "task": {
                "type": "generate_dialogue",
                "scene_id": "scene_1",
                "characters": ["character_1", "character_2"]
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_dialogue_generation(mock_state):
    """Test dialogue generation functionality."""
    crafter = DialogueCrafter()
    result = await crafter.process(mock_state, {})
    
    assert "messages" in result
    assert "dialogue_updates" in result
    assert "dialogue" in result["dialogue_updates"]

@pytest.mark.asyncio
async def test_character_voice(mock_state):
    """Test character voice definition."""
    mock_state.current_input["task"]["type"] = "define_voice"
    crafter = DialogueCrafter()
    result = await crafter.process(mock_state, {})
    
    assert "messages" in result
    assert "dialogue_updates" in result
    assert "voices" in result["dialogue_updates"]

@pytest.mark.asyncio
async def test_subtext_analysis(mock_state):
    """Test dialogue subtext analysis."""
    mock_state.current_input["task"]["type"] = "analyze_subtext"
    crafter = DialogueCrafter()
    result = await crafter.process(mock_state, {})
    
    assert "messages" in result
    assert "dialogue_updates" in result
    assert "subtext" in result["dialogue_updates"]
