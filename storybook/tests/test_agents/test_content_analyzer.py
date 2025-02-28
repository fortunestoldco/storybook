import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.agents.content_analyzer import ContentAnalyzer


class TestContentAnalyzer:
    @pytest.fixture
    def content_analyzer(self, mock_llm, mock_document_store):
        return ContentAnalyzer()

    def test_analyze_content(self, content_analyzer, mock_document_store):
        manuscript_id = "test_123"

        result = content_analyzer.analyze_content(manuscript_id)

        assert isinstance(result, dict)
        assert "analysis" in result
        mock_document_store.get_manuscript.assert_called_once_with(manuscript_id)
