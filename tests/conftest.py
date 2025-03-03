import pytest
from typing import Dict, Any
from mongodb import MongoDBManager

@pytest.fixture
def mongodb_manager():
    """Fixture for MongoDB manager."""
    return MongoDBManager()

@pytest.fixture
def test_input() -> Dict[str, Any]:
    """Fixture for test input data."""
    return {
        "title": "Test Story",
        "manuscript": "Test content",
        "model_provider": "anthropic",
        "model_name": "claude-3-opus-20240229"
    }