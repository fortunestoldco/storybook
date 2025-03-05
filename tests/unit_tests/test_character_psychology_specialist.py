import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.character_psychology_specialist import CharacterPsychologySpecialist
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
                    "motivation": {},
                    "conflict_response": {}
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
                "type": "analyze_motivation",
                "character_id": "char_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_motivation_analysis(mock_state):
    """Test character motivation analysis."""
    specialist = CharacterPsychologySpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "character_updates" in result
    assert "motivation" in result["character_updates"]["char_1"]

@pytest.mark.asyncio
async def test_conflict_response(mock_state):
    """Test character conflict response analysis."""
    mock_state.current_input["task"]["type"] = "analyze_conflict"
    specialist = CharacterPsychologySpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "character_updates" in result
    assert "conflict_response" in result["character_updates"]["char_1"]

@pytest.mark.asyncio
async def test_psychology_profile(mock_state):
    """Test character psychology profile update."""
    mock_state.current_input["task"]["type"] = "update_profile"
    specialist = CharacterPsychologySpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "character_updates" in result
    assert "profile" in result["character_updates"]["char_1"]
