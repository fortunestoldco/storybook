import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
from datetime import datetime

from storybook.configuration import Configuration
from storybook.state import NovelSystemState, Project
from storybook.graph import create_storybook_graph
from storybook.agents.factory import AgentFactory
from storybook.tools.quality import QualityMetricsTool, QualityGateTool

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        genre=["Fantasy"],
        target_audience=["Young Adult"],
        content={},
        quality_assessment={
            "plot_coherence": 0.8,
            "character_development": 0.9,
            "world_building": 0.85,
            "prose_quality": 0.75
        }
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="initialization",
        project=mock_project,
        current_input={"task": "initial_quality_assessment"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_full_initialization_phase(mock_state):
    """Test the initialization phase execution."""
    graph = create_storybook_graph({"configurable": {"project_id": "test"}})
    result = await graph.ainvoke(mock_state)
    
    assert result.phase == "initialization"
    assert "quality_assessment" in result.project.quality_assessment
    assert isinstance(result.phase_history, dict)

@pytest.mark.asyncio
async def test_phase_transition(mock_project):
    """Test phase transition logic and quality gates."""
    config = Configuration(
        model="gpt-4",
        agent_roles={
            "executive_director": "Executive Director role",
            "quality_assessment_director": "Quality Assessment Director role"
        },
        quality_gates={
            "initialization_to_development": {
                "plot_coherence": 0.7,
                "character_development": 0.7,
                "world_building": 0.7
            }
        }
    )
    
    state = NovelSystemState(
        phase="initialization",
        project=mock_project,
        current_input={"task": "evaluate_phase_completion"},
        phase_history={}
    )
    
    graph = create_storybook_graph({
        "configurable": {
            "project_id": "test",
            "quality_gates": config.quality_gates
        }
    })
    
    result = await graph.ainvoke(state)
    
    # Should transition to development phase since quality metrics exceed gates
    assert result.phase == "development"
    assert "initialization" in result.phase_history
    assert result.phase_history["initialization"][-1]["transition_to"] == "development"

@pytest.mark.asyncio
async def test_tool_chain():
    """Test multiple tools working together in sequence."""
    mock_content = {
        "plot_threads": [{"id": "main_plot", "description": "Hero's journey"}],
        "characters": {
            "protagonist": {"name": "Hero", "arc": "growth"}
        }
    }
    
    state = NovelSystemState(
        phase="development",
        project=Project(
            title="Test Novel",
            genre=["Fantasy"],
            content=mock_content,
            quality_assessment={}
        ),
        current_input={"task": "develop_plot"},
        phase_history={}
    )
    
    with patch('storybook.tools.plot.PlotThreadTool._arun') as mock_plot_tool, \
         patch('storybook.tools.character.CharacterArcTool._arun') as mock_arc_tool:
        
        mock_plot_tool.return_value = {"plot_update": {"thread_id": "main_plot"}}
        mock_arc_tool.return_value = {"arc_update": {"character_id": "protagonist"}}
        
        graph = create_storybook_graph({"configurable": {"project_id": "test"}})
        result = await graph.ainvoke(state)
        
        assert "plot_update" in result.project.content
        assert "arc_update" in result.project.content
        assert mock_plot_tool.called
        assert mock_arc_tool.called

@pytest.mark.asyncio
async def test_phase_transitions():
    """Test phase transition logic and state maintenance."""
    phases = ["initialization", "development", "creation", "refinement", "finalization"]
    mock_content = {"initial_content": "test"}
    
    state = NovelSystemState(
        phase=phases[0],
        project=Project(
            title="Test Novel",
            genre=["Fantasy"],
            content=mock_content,
            quality_assessment={"overall": 0.9}
        ),
        current_input={"task": "phase_transition"},
        phase_history={}
    )
    
    graph = create_storybook_graph({
        "configurable": {
            "project_id": "test",
            "quality_gates": {phase: {"overall": 0.8} for phase in phases[:-1]}
        }
    })
    
    for expected_next_phase in phases[1:]:
        result = await graph.ainvoke(state)
        assert result.phase == expected_next_phase
        assert result.project.content["initial_content"] == "test"
        assert isinstance(result.phase_history.get(state.phase), list)
        state = result

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in the graph execution."""
    state = NovelSystemState(
        phase="development",
        project=Project(
            title="Test Novel",
            genre=["Fantasy"],
            content={},
            quality_assessment={}
        ),
        current_input={"task": "invalid_task"},
        phase_history={}
    )
    
    with patch('storybook.tools.base.NovelWritingTool._arun') as mock_tool:
        mock_tool.side_effect = Exception("Tool execution failed")
        
        graph = create_storybook_graph({"configurable": {"project_id": "test"}})
        result = await graph.ainvoke(state)
        
        assert result.phase == "development"  # Phase shouldn't change on error
        assert "error" in result.project.content
        assert result.project.content["error"]["type"] == "tool_execution"

@pytest.mark.asyncio
async def test_cross_phase_consistency():
    """Test consistency of state and content across phase transitions."""
    config = Configuration(
        model="gpt-4",
        agent_roles={
            "executive_director": "Executive Director role",
            "creative_director": "Creative Director role"
        },
        quality_gates={
            "initialization_to_development": {"minimum_quality": 0.7}
        }
    )
    
    initial_content = {
        "plot_outline": "Basic plot structure",
        "world_building": {"setting": "Fantasy world"}
    }
    
    state = NovelSystemState(
        phase="initialization",
        project=Project(
            title="Test Novel",
            genre=["Fantasy"],
            content=initial_content,
            quality_assessment={"minimum_quality": 0.8}
        ),
        current_input={"task": "complete_phase"}
    )
    
    graph = create_storybook_graph({
        "configurable": {
            "project_id": "test",
            "quality_gates": config.quality_gates
        }
    })
    
    # Test transition through phases
    result = await graph.ainvoke(state)
    assert result.phase == "development"
    assert result.project.content["plot_outline"] == initial_content["plot_outline"]
    assert result.project.content["world_building"] == initial_content["world_building"]