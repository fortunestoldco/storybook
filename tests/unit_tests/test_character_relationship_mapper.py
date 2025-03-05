import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.character_relationship_mapper import CharacterRelationshipMapper
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    return Project(
        title="Test Novel",
        content={
            "characters": {
                "char_1": {"name": "Alice"},
                "char_2": {"name": "Bob"}
            },
            "relationship_graph": {},
            "plot_threads": []
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    return NovelSystemState(
        phase="development",
        project=mock_project,
        current_input={"task": "analyze_dynamics"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_dynamics_analysis(mock_state):
    """Test character dynamics analysis."""
    mapper = CharacterRelationshipMapper()
    result = await mapper.process(mock_state, {})
    
    assert "messages" in result
    assert "relationship_updates" in result
    assert "dynamics" in result["relationship_updates"]

@pytest.mark.asyncio
async def test_conflict_mapping(mock_state):
    """Test character conflict mapping."""
    mock_state.current_input["task"] = "map_conflict"
    mapper = CharacterRelationshipMapper()
    result = await mapper.process(mock_state, {})
    
    assert "messages" in result
    assert "relationship_updates" in result
    assert "conflicts" in result["relationship_updates"]

@pytest.mark.asyncio
async def test_relationship_graph_update(mock_state):
    """Test relationship graph update."""
    mock_state.current_input["task"] = "update_graph"
    mapper = CharacterRelationshipMapper()
    result = await mapper.process(mock_state, {})
    
    assert "messages" in result
    assert "relationship_updates" in result
    assert "graph" in result["relationship_updates"]
