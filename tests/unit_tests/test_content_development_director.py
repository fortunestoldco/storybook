import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.content_development_director import ContentDevelopmentDirector
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    return Project(
        title="Test Novel",
        content={
            "outline": {},
            "milestones": {},
            "quality_criteria": {}
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    return NovelSystemState(
        phase="creation",
        project=mock_project,
        current_input={"task": "plan_content"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_content_planning(mock_state):
    """Test content planning functionality."""
    director = ContentDevelopmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "content_updates" in result
    assert "plan" in result["content_updates"]

@pytest.mark.asyncio
async def test_quality_assessment(mock_state):
    """Test content quality assessment."""
    mock_state.current_input["task"] = "assess_quality"
    director = ContentDevelopmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "content_updates" in result
    assert "quality" in result["content_updates"]

@pytest.mark.asyncio
async def test_progress_tracking(mock_state):
    """Test content progress tracking."""
    mock_state.current_input["task"] = "track_progress"
    director = ContentDevelopmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "content_updates" in result
    assert "progress" in result["content_updates"]