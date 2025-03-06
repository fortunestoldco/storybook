import pytest
from langchain_core.tools import BaseTool
from storybook.tools import ToolRegistry

class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool for testing"
    
    def _run(self):
        return "mock result"

@pytest.fixture
def tool_registry():
    return ToolRegistry()

@pytest.fixture
def mock_tool():
    return MockTool()

def test_register_tool(tool_registry, mock_tool):
    # Act
    tool_registry.register_tool(mock_tool, "test_agent")
    
    # Assert
    tools = tool_registry.get_tools_for_agent("test_agent")
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"

def test_register_multiple_tools(tool_registry):
    # Arrange
    tools = [MockTool(), MockTool()]
    
    # Act
    tool_registry.register_tools(tools, "test_agent")
    
    # Assert
    registered_tools = tool_registry.get_tools_for_agent("test_agent")
    assert len(registered_tools) == 2

def test_get_nonexistent_agent_tools(tool_registry):
    # Act
    tools = tool_registry.get_tools_for_agent("nonexistent_agent")
    
    # Assert
    assert isinstance(tools, list)
    assert len(tools) == 0