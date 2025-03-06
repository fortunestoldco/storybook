import pytest
from unittest.mock import Mock, patch
from storybook.tools.management_tools import project_management_tool, timeline_management_tool
from storybook.configuration import Configuration

@pytest.fixture
def mock_db():
    with patch('storybook.db_config.get_collection') as mock:
        collection = Mock()
        collection.find_one.return_value = {"_id": "test_id", "name": "test"}
        collection.insert_one.return_value.inserted_id = "new_id"
        mock.return_value = collection
        yield mock

@pytest.fixture
def config():
    return Configuration()

def test_project_management_tool(mock_db, config):
    result = project_management_tool(
        action="create",
        project_id="test_project",
        project_data={"name": "Test Project"}
    )
    assert result["status"] == "success"
    mock_db.assert_called_once_with(COLLECTIONS["projects"])

def test_timeline_management_tool(mock_db):
    result = timeline_management_tool(
        action="create",
        project_id="test_project",
        timeline_data={"name": "Test Timeline"}
    )
    assert result["status"] == "success"
    mock_db.assert_called_once_with(COLLECTIONS["timelines"])