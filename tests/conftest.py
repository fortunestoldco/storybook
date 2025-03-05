import pytest
import os
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'LANGCHAIN_API_KEY': 'test_key',
        'LANGCHAIN_ENDPOINT': 'http://localhost:8000',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        yield

@pytest.fixture
def mock_content():
    """Provide mock content for testing."""
    return {
        "chapter": "Test content",
        "metadata": {
            "word_count": 1000,
            "chapter_number": 1
        }
    }