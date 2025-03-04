import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from storybook.configuration import Configuration
from storybook.state import NovelSystemState
from storybook.agents.factory import AgentFactory

@pytest.fixture
def mock_config():
    return Configuration(
        model="gpt-4",
        agent_roles={
            "executive_director": "Executive Director role",
            "creative_director": "Creative Director role"
        },
        quality_gates={}
    )

@pytest.fixture
def agent_factory(mock_config):
    return AgentFactory(mock_config)

@pytest.mark.asyncio
async def test_agent_factory_initialization(agent_factory):
    assert agent_factory.config is not None
    assert agent_factory.base_model is not None
    assert hasattr(agent_factory, 'tool_registry')

@pytest.mark.asyncio
async def test_agent_creation(agent_factory):
    agent_func = agent_factory.create_agent("executive_director", "test_project")
    assert callable(agent_func)

@pytest.mark.asyncio
async def test_agent_execution(agent_factory):
    state = NovelSystemState(
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
async def test_tool_execution():
    with patch('storybook.tools.quality.QualityMetricsTool._arun') as mock_tool:
        mock_tool.return_value = {"quality_metrics": {"score": 0.8}}
        
        # Test tool execution
        tool = QualityMetricsTool()
        result = await tool._arun(content={}, metrics={})
        
        assert "quality_metrics" in result
        assert result["quality_metrics"]["score"] == 0.8