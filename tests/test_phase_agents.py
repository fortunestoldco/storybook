import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from storybook.state import NovelSystemState, Project
from storybook.configuration import Configuration
from storybook.agents.factory import AgentFactory

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        genre=["Fantasy"],
        target_audience=["Young Adult"],
        content={},
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="initialization",
        project=mock_project,
        current_input={"task": "assess_quality"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_initialization_phase_agents(mock_state):
    """Test initialization phase agents."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={
            "executive_director": "Executive Director role",
            "quality_assessment_director": "Quality Assessment Director role"
        }
    )
    factory = AgentFactory(config)
    
    for agent_name in ["executive_director", "quality_assessment_director"]:
        agent_func = factory.create_agent(agent_name, "test_project")
        assert callable(agent_func)
        
        result = await agent_func(mock_state, {"configurable": {}})
        assert isinstance(result, dict)
        assert "messages" in result

@pytest.mark.asyncio
async def test_development_phase_agents(mock_state):
    """Test development phase agents."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={
            "creative_director": "Creative Director role",
            "world_building_expert": "World Building Expert role",
            "plot_development_specialist": "Plot Development Specialist role",
            "structure_architect": "Structure Architect role"
        }
    )
    factory = AgentFactory(config)
    
    mock_state.phase = "development"
    for agent_name in config.agent_roles.keys():
        agent_func = factory.create_agent(agent_name, "test_project")
        assert callable(agent_func)
        
        result = await agent_func(mock_state, {"configurable": {}})
        assert isinstance(result, dict)
        assert "messages" in result

@pytest.mark.asyncio
async def test_creation_phase_agents(mock_state):
    """Test creation phase agents."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={
            "content_development_director": "Content Development Director role",
            "chapter_drafter": "Chapter Drafter role",
            "dialogue_crafter": "Dialogue Crafter role"
        }
    )
    factory = AgentFactory(config)
    
    mock_state.phase = "creation"
    for agent_name in config.agent_roles.keys():
        agent_func = factory.create_agent(agent_name, "test_project")
        assert callable(agent_func)
        
        result = await agent_func(mock_state, {"configurable": {}})
        assert isinstance(result, dict)
        assert "messages" in result
        assert result.get("error") is None

@pytest.mark.asyncio
async def test_agent_tool_execution(mock_state):
    """Test agent tool execution."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={"executive_director": "Executive Director role"}
    )
    factory = AgentFactory(config)
    
    with patch('storybook.tools.quality.QualityMetricsTool._arun') as mock_tool:
        mock_tool.return_value = {"quality_metrics": {"score": 0.8}}
        
        agent_func = factory.create_agent("executive_director", "test_project")
        result = await agent_func(mock_state, {"configurable": {}})
        
        assert mock_tool.called
        assert "quality_metrics" in result
        assert result["quality_metrics"]["score"] == 0.8

@pytest.mark.asyncio
async def test_agent_error_handling(mock_state):
    """Test agent error handling."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={"executive_director": "Executive Director role"}
    )
    factory = AgentFactory(config)
    
    with patch('storybook.tools.quality.QualityMetricsTool._arun') as mock_tool:
        mock_tool.side_effect = Exception("Test error")
        
        agent_func = factory.create_agent("executive_director", "test_project")
        result = await agent_func(mock_state, {"configurable": {}})
        
        assert "error" in result
        assert "messages" in result
        assert "Test error" in str(result["error"])