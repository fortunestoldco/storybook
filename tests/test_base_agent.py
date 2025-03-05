import pytest
from unittest.mock import Mock, patch
from storybook.agents.base_agent import BaseAgent
from storybook.state import NovelSystemState

class TestAgent(BaseAgent):
    """Test agent implementation."""
    async def process(self, state: NovelSystemState, config: Dict) -> Dict[str, Any]:
        return {"test": "result"}

@pytest.mark.asyncio
async def test_base_agent_initialization():
    """Test base agent initialization."""
    agent = TestAgent("test_agent", [Mock()])
    assert agent.name == "test_agent"
    assert len(agent.tools) == 1

@pytest.mark.asyncio
async def test_base_agent_system_prompt():
    """Test base agent system prompt generation."""
    agent = TestAgent("test_agent", [])
    state = Mock(project_id="test_project")
    prompt = agent.get_system_prompt(state)
    assert isinstance(prompt, SystemMessage)