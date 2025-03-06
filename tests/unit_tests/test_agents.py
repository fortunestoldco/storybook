import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from storybook.agents import (
    ExecutiveDirector,
    CreativeDirector,
    EditorialDirector
)

@pytest.fixture
def mock_tool_registry():
    registry = Mock()
    registry.get_tools_for_agent.return_value = []
    return registry

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Test response")
    return llm

def test_executive_director_initialization(mock_tool_registry, mock_llm):
    # Act
    agent = ExecutiveDirector(
        tool_registry=mock_tool_registry,
        llm=mock_llm
    )
    
    # Assert
    assert agent.name == "executive_director"
    mock_tool_registry.get_tools_for_agent.assert_called_with("executive_director")

@pytest.mark.asyncio
async def test_executive_director_invoke(mock_tool_registry, mock_llm):
    # Arrange
    agent = ExecutiveDirector(
        tool_registry=mock_tool_registry,
        llm=mock_llm
    )
    
    # Act
    result = await agent.invoke({"input": "test input"})
    
    # Assert
    assert isinstance(result, dict)
    assert "output" in result

def test_creative_director_initialization(mock_tool_registry, mock_llm):
    # Act
    agent = CreativeDirector(
        tool_registry=mock_tool_registry,
        llm=mock_llm
    )
    
    # Assert
    assert agent.name == "creative_director"
    mock_tool_registry.get_tools_for_agent.assert_called_with("creative_director")