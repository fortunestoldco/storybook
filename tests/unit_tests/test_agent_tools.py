import pytest
from unittest.mock import Mock, patch
from storybook.tools.management_tools import (
    ProjectTimelineTool,
    TeamCommunicationTool,
    ResourceAllocationTool
)

def test_project_timeline_tool():
    # Arrange
    tool = ProjectTimelineTool()
    
    # Act
    result = tool.run("Check project timeline")
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0

@patch('storybook.tools.management_tools.send_team_message')
def test_team_communication_tool(mock_send):
    # Arrange
    tool = TeamCommunicationTool()
    mock_send.return_value = True
    
    # Act
    result = tool.run("Send message to team")
    
    # Assert
    assert isinstance(result, str)
    mock_send.assert_called_once()

def test_resource_allocation_tool():
    # Arrange
    tool = ResourceAllocationTool()
    
    # Act
    result = tool.run("Allocate resources for chapter 1")
    
    # Assert
    assert isinstance(result, str)
    assert "resources" in result.lower()