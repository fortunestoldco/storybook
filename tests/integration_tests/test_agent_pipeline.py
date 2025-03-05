import pytest
from typing import Dict, Any
from unittest.mock import Mock

from storybook.state import NovelSystemState
from storybook.configuration import Configuration
from storybook.agents.factory import AgentFactory

@pytest.fixture
def config():
    return Configuration(
        model="test-model",
        agents={
            "executive_director": {"enabled": True},
            "creative_director": {"enabled": True}
        }
    )

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="initialization",
        project=Mock(
            content={
                "outline": {},
                "characters": {},
                "settings": {}
            }
        ),
        current_input={},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_agent_pipeline():
    """Test complete agent pipeline."""
    factory = AgentFactory(config)
    
    # Test initialization phase
    executive_director = factory.create_agent("executive_director")
    result = await executive_director.process(mock_state, {})
    assert "messages" in result
    
    # Test development phase
    mock_state.phase = "development"
    creative_director = factory.create_agent("creative_director")
    result = await creative_director.process(mock_state, {})
    assert "messages" in result
    
    # Test creation phase
    mock_state.phase = "creation"
    content_director = factory.create_agent("content_development_director")
    result = await content_director.process(mock_state, {})
    assert "messages" in result