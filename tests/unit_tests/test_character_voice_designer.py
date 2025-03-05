import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.character_voice_designer import CharacterVoiceDesigner
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "characters": {
                "char_1": {
                    "profile": {},
                    "voice_pattern": {}
                }
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
                "type": "update_dialogue_style",
                "character_id": "char_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_dialogue_style_update(mock_state):
    """Test dialogue style update."""
    designer = CharacterVoiceDesigner()
    result = await designer.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "dialogue_style" in result["voice_updates"]["char_1"]

@pytest.mark.asyncio
async def test_expression_analysis(mock_state):
    """Test expression analysis."""
    mock_state.current_input["task"]["type"] = "analyze_expression"
    designer = CharacterVoiceDesigner()
    result = await designer.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "expression_analysis" in result["voice_updates"]["char_1"]

@pytest.mark.asyncio
async def test_voice_pattern_update(mock_state):
    """Test voice pattern update."""
    mock_state.current_input["task"]["type"] = "update_voice_pattern"
    designer = CharacterVoiceDesigner()
    result = await designer.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "voice_pattern" in result["voice_updates"]["char_1"]
