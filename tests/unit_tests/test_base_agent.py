from typing import Dict, Any
import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import SystemMessage

from storybook.agents.base_agent import BaseAgent
from storybook.state import NovelSystemState
from storybook.tools.base import NovelWritingTool

class MockTool(NovelWritingTool):
    name = "mock_tool"
    description = "Mock tool for testing"
    
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return {"result": "mock_result"}

class TestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="test_agent",
            tools=[MockTool()]
        )
    
    async def _process(self, state: NovelSystemState, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [],
            "updates": {"test": "value"}
        }

@pytest.fixture
def agent():
    return TestAgent()

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="test",
        project=Mock(content={}),
        current_input={},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent.name == "test_agent"
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], NovelWritingTool)

@pytest.mark.asyncio
async def test_agent_process(agent, mock_state):
    """Test agent processing."""
    result = await agent.process(mock_state, {})
    assert "messages" in result
    assert "updates" in result
    assert result["updates"]["test"] == "value"

@pytest.mark.asyncio
async def test_agent_error_handling(agent, mock_state):
    """Test agent error handling."""
    with patch.object(agent, '_process', side_effect=Exception("Test error")):
        result = await agent.process(mock_state, {})
        assert "error" in result
        assert "messages" in result
        assert "failed" in result["status"]