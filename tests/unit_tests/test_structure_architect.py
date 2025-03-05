import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.structure_architect import StructureArchitect
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "structure": {},
            "outline": {}
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
            "task": "update_outline"
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_outline_update(mock_state):
    """Test chapter outline update."""
    architect = StructureArchitect()
    result = await architect.process(mock_state, {})
    
    assert "messages" in result
    assert "structure_updates" in result
    assert "outline" in result["structure_updates"]

@pytest.mark.asyncio
async def test_structure_analysis(mock_state):
    """Test structure analysis."""
    mock_state.current_input["task"] = "analyze_structure"
    architect = StructureArchitect()
    result = await architect.process(mock_state, {})
    
    assert "messages" in result
    assert "structure_updates" in result
    assert "analysis" in result["structure_updates"]
