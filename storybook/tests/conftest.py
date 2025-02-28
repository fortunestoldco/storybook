import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from storybook.api import app
from storybook.db.document_store import DocumentStore
from storybook.config import get_llm


@pytest.fixture
def mock_document_store():
    with patch("storybook.db.document_store.DocumentStore") as mock_store:
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.store_manuscript.return_value = "test_id"
        mock_store_instance.get_manuscript.return_value = {
            "title": "Test Manuscript",
            "content": "Test content",
            "metadata": {"author": "Test Author"},
        }
        yield mock_store_instance


@pytest.fixture
def mock_llm():
    with patch("storybook.config.get_llm") as mock_llm:
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value = "Test LLM response"
        yield mock_llm_instance


@pytest.fixture
def test_client():
    return TestClient(app)
