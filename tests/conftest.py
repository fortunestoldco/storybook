import pytest
from typing import Dict, Any
from mongodb import MongoDBManager
from agents import AgentFactory
from config import MONGODB_CONFIG

@pytest.fixture
def mongodb_manager():
    """Fixture for MongoDB manager."""
    return MongoDBManager()

@pytest.fixture
def agent_factory(mongodb_manager):
    """Fixture for agent factory."""
    return AgentFactory(mongodb_manager)

@pytest.fixture
def test_input() -> Dict[str, Any]:
    """Fixture for test input data."""
    return {
        "title": "Test Story",
        "manuscript": "Test content",
        "model_provider": "anthropic",
        "model_name": "claude-3-opus-20240229"
    }