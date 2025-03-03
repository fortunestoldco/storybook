import pytest
from unittest.mock import patch, MagicMock
from mongodb import MongoDBManager
from config import MONGODB_CONFIG

@pytest.fixture
def mock_client():
    with patch('mongodb.MongoClient') as mock:
        yield mock

@pytest.fixture
def mongodb_manager(mock_client):
    manager = MongoDBManager()
    manager.client = mock_client
    return manager

def test_save_document(mongodb_manager, mock_client):
    """Test document saving functionality."""
    test_doc = {"_id": "test123", "content": "test"}
    mongodb_manager.save_document(test_doc)
    mock_client[MONGODB_CONFIG["database_name"]].documents.replace_one.assert_called_once()

def test_save_state(mongodb_manager, mock_client):
    project_id = "test_project"
    state = {"key": "value"}
    mongodb_manager.save_state(project_id, state)
    collection = mock_client[MONGODB_CONFIG["database_name"]][MONGODB_CONFIG["collections"]["project_state"]]
    collection.update_one.assert_called_once_with(
        {"project_id": project_id},
        {"$set": state},
        upsert=True
    )

def test_load_state(mongodb_manager, mock_client):
    project_id = "test_project"
    expected_state = {"project_id": project_id, "key": "value"}
    collection = mock_client[MONGODB_CONFIG["database_name"]][MONGODB_CONFIG["collections"]["project_state"]]
    collection.find_one.return_value = expected_state
    state = mongodb_manager.load_state(project_id)
    assert state == expected_state

def test_load_document(mongodb_manager, mock_client):
    document_id = "test_id"
    expected_document = {"_id": document_id, "key": "value"}
    collection = mock_client[MONGODB_CONFIG["database_name"]][MONGODB_CONFIG["collections"]["documents"]]
    collection.find_one.return_value = expected_document
    document = mongodb_manager.load_document(document_id)
    assert document == expected_document

def test_save_research(mongodb_manager, mock_client):
    research = {"_id": "test_id", "key": "value"}
    mongodb_manager.save_research(research)
    collection = mock_client[MONGODB_CONFIG["database_name"]][MONGODB_CONFIG["collections"]["research"]]
    collection.replace_one.assert_called_once_with(
        {"_id": research["_id"]}, research, upsert=True
    )
