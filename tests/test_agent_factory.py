import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
import os

from storybook.configuration import Configuration
from storybook.agents.factory import AgentFactory

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'LANGCHAIN_API_KEY': 'test_key',
        'LANGCHAIN_ENDPOINT': 'http://localhost:8000',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        yield

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Configuration(
        model="openai/gpt-4",
        agent_roles={
            "executive_director": "Executive Director role",
            "creative_director": "Creative Director role"
        },
        quality_gates={}
    )

@pytest.fixture
async def agent_factory(mock_config, mock_env_vars):
    """Create an agent factory for testing."""
    with patch('storybook.utils.init_chat_model') as mock_init_model:
        mock_init_model.return_value = Mock()
        factory = AgentFactory(mock_config)
        return factory

@pytest.mark.asyncio
async def test_agent_factory_initialization(agent_factory):
    """Test agent factory initialization."""
    assert agent_factory.config is not None
    assert agent_factory.base_model is not None
    assert hasattr(agent_factory, 'tool_registry')

@pytest.mark.asyncio
async def test_agent_creation(agent_factory):
    """Test agent creation."""
    agent_func = agent_factory.create_agent("executive_director", "test_project")
    assert callable(agent_func)

@pytest.mark.asyncio
async def test_agent_execution(agent_factory):
    """Test agent execution."""
    with patch('storybook.tools.quality.QualityAssessmentTool._arun') as mock_tool:
        mock_tool.return_value = {"quality_assessment": {"score": 0.8}}
        
        state = Mock(
            phase="initialization",
            project=Mock(content={}, quality_assessment={}),
            current_input={"task": "quality_check"}
        )
        config = {"configurable": {}}
        
        agent_func = agent_factory.create_agent("executive_director", "test_project")
        result = await agent_func(state, config)
        
        assert isinstance(result, dict)
        assert "messages" in result

@pytest.mark.asyncio
async def test_agent_tool_registry():
    """Test agent tool registry initialization."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={"executive_director": "Executive Director role"}
    )
    factory = AgentFactory(config)
    
    assert "executive_director" in factory.tool_registry
    assert len(factory.tool_registry["executive_director"]) > 0
    assert all(isinstance(tool, NovelWritingTool) for tool in factory.tool_registry["executive_director"])

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling."""
    with patch('storybook.tools.quality.QualityMetricsTool._arun') as mock_tool:
        mock_tool.side_effect = Exception("Test error")
        
        state = Mock(
            phase="initialization",
            project=Mock(content={}, quality_assessment={}),
            current_input={"task": "quality_check"}
        )
        
        result = await factory._execute_agent(
            agent_name="test_agent",
            tools=[QualityMetricsTool()],
            state=state,
            system_prompt="test prompt",
            config=Mock()
        )
        
        assert "error" in result
        assert "messages" in result
        assert "Test error" in str(result["error"])

@pytest.mark.asyncio
async def test_invalid_agent_creation():
    """Test creating agent with invalid role."""
    config = Configuration(
        model="openai/gpt-4",
        agent_roles={"executive_director": "Executive Director role"}
    )
    factory = AgentFactory(config)
    
    with pytest.raises(ValueError, match="Unknown agent role"):
        factory.create_agent("invalid_role", "test_project")

@pytest.mark.asyncio
async def test_agent_state_updates():
    """Test agent state updates."""
    state = Mock(
        phase="initialization",
        project=Mock(
            content={"test": "content"},
            quality_assessment={}
        ),
        current_input={"task": "quality_check"},
        messages=[],
        agent_outputs={}
    )
    
    agent_func = factory.create_agent("executive_director", "test_project")
    result = await agent_func(state, {"configurable": {}})
    
    assert "messages" in result
    assert "current_agent" in result
    assert "agent_outputs" in result