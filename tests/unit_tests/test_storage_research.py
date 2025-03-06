import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from storybook.storage.research import ResearchStorage

@pytest.fixture
def mock_collection():
    with patch('storybook.db_config.get_collection') as mock:
        collection = Mock()
        collection.find_one.return_value = {"_id": "test_id", "data": {"test": "data"}}
        collection.insert_one.return_value.inserted_id = "new_id"
        mock.return_value = collection
        yield collection

@pytest.fixture
def storage():
    return ResearchStorage("test_project")

@pytest.mark.asyncio
async def test_store_research(storage, mock_collection):
    data = {"test": "data"}
    result = await storage.store_research(data)
    
    assert result == "new_id"
    mock_collection.insert_one.assert_called_once()

@pytest.mark.asyncio
async def test_get_research(storage, mock_collection):
    result = await storage.get_research("test_id")
    
    assert result["data"]["test"] == "data"
    mock_collection.find_one.assert_called_once_with({"_id": "test_id"})

@pytest.mark.asyncio
async def test_list_research(storage, mock_collection):
    mock_collection.find.return_value = [
        {"_id": "1", "data": {"test": "data1"}},
        {"_id": "2", "data": {"test": "data2"}}
    ]
    
    result = await storage.list_research()
    
    assert len(result) == 2
    mock_collection.find.assert_called_once()

@pytest.mark.asyncio
async def test_update_research(storage, mock_collection):
    mock_collection.update_one.return_value.modified_count = 1
    data = {"test": "updated"}
    
    result = await storage.update_research("test_id", data)
    
    assert result is True
    mock_collection.update_one.assert_called_once()