import pytest
from unittest.mock import Mock, patch
from pymongo import MongoClient
from storybook.db_config import (
    initialize_config,
    get_client,
    get_database,
    get_collection,
    close_connection,
    COLLECTIONS
)
from storybook.configuration import Configuration

@pytest.fixture
def mock_config():
    config = Configuration()
    config.mongodb_connection_string = "mongodb://localhost:27017"
    config.mongodb_database_name = "test_db"
    return config

@pytest.fixture
def initialized_db(mock_config):
    initialize_config(mock_config)
    yield
    close_connection()

def test_initialize_config(mock_config):
    initialize_config(mock_config)
    with patch('pymongo.MongoClient') as mock_client:
        client = get_client()
        assert client is not None

def test_get_database(initialized_db):
    with patch('pymongo.MongoClient') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        db = get_database()
        assert db is not None

def test_get_collection(initialized_db):
    with patch('pymongo.MongoClient') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        collection = get_collection(COLLECTIONS["projects"])
        assert collection is not None

def test_close_connection(initialized_db):
    with patch('pymongo.MongoClient') as mock_client:
        client = get_client()
        close_connection()
        assert client is None