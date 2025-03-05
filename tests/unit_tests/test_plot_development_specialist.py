import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.plot_development_specialist import PlotDevelopmentSpecialist
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    return Project(
        title="Test Novel",
        content={
            "plot": {}
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    return NovelSystemState(
        phase="development",
        project=mock_project,
        current_input={"task": "update_plot_structure"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_plot_structure_update(mock_state):
    """Test plot structure update functionality."""
    specialist = PlotDevelopmentSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "plot_updates" in result
    assert "structure" in result["plot_updates"]

@pytest.mark.asyncio
async def test_conflict_development(mock_state):
    """Test conflict development functionality."""
    mock_state.current_input["task"] = "update_conflict_development"
    specialist = PlotDevelopmentSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "plot_updates" in result
    assert "conflicts" in result["plot_updates"]
