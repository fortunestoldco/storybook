import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from storybook.tools.base import NovelWritingTool
from storybook.agents.base_agent import BaseAgent
from storybook.state import NovelSystemState

class ErrorTool(NovelWritingTool):
    name = "error_tool"
    description = "Tool that raises errors"
    
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        raise ValueError("Test error")

class ErrorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="error_agent",
            tools=[ErrorTool()]
        )
    
    async def _process(self, state: NovelSystemState, config: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.tools[0]._arun()
        return {"result": result}

@pytest.fixture
def error_agent():
    return ErrorAgent()

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="test",
        project=Mock(content={}),
        current_input={},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_tool_error_handling(error_agent, mock_state):
    """Test tool error handling."""
    result = await error_agent.process(mock_state, {})
    assert "error" in result
    assert "Test error" in result["error"]
    assert result["status"] == "failed"