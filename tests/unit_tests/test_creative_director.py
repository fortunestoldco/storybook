import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.creative_director import CreativeDirector
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    return Project(
        title="Test Novel",
        content={
            "outline": {},
            "characters": {},
            "plot": {}
        },
        style_preferences={}
    )

@pytest.fixture
def mock_state(mock_project):
    return NovelSystemState(
        phase="development",
        project=mock_project,
        current_input={"task": "assess_creative_elements"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_creative_assessment(mock_state):
    """Test creative assessment functionality."""
    director = CreativeDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "creative_updates" in result
    assert "assessment" in result["creative_updates"]

@pytest.mark.asyncio
async def test_story_elements_update(mock_state):
    """Test story elements update functionality."""
    mock_state.current_input["task"] = "update_story_elements"
    director = CreativeDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "creative_updates" in result
    assert "story_elements" in result["creative_updates"]

@pytest.mark.asyncio
async def test_creative_vision_update(mock_state):
    """Test creative vision update functionality."""
    mock_state.current_input["task"] = "update_creative_vision"
    director = CreativeDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "creative_updates" in result
    assert "vision" in result["creative_updates"]
