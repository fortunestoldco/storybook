import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.creation.chapter_drafter import ChapterDrafter
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "outline": {},
            "scenes": {},
            "narrative_flow": {}
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="creation",
        project=mock_project,
        current_input={
            "task": {
                "type": "structure",
                "chapter_id": "chapter_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_chapter_structure(mock_state):
    """Test chapter structure creation."""
    drafter = ChapterDrafter()
    result = await drafter.process(mock_state, {})
    
    assert "messages" in result
    assert "chapter_updates" in result
    assert "structure" in result["chapter_updates"]

@pytest.mark.asyncio
async def test_scene_sequence(mock_state):
    """Test scene sequence optimization."""
    mock_state.current_input["task"]["type"] = "sequence"
    drafter = ChapterDrafter()
    result = await drafter.process(mock_state, {})
    
    assert "messages" in result
    assert "chapter_updates" in result
    assert "sequence" in result["chapter_updates"]

@pytest.mark.asyncio
async def test_narrative_flow(mock_state):
    """Test narrative flow analysis."""
    mock_state.current_input["task"]["type"] = "flow"
    drafter = ChapterDrafter()
    result = await drafter.process(mock_state, {})
    
    assert "messages" in result
    assert "chapter_updates" in result
    assert "flow" in result["chapter_updates"]
