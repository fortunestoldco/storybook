import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.agents.world_builder import WorldBuilder


class TestWorldBuilder:
    @pytest.fixture
    def world_builder(self, mock_llm, mock_document_store):
        return WorldBuilder()

    def test_build_world(self, world_builder, mock_document_store):
        manuscript_id = "test_123"
        target_audience = {"demographic": "Young Adult"}
        research_insights = {"genre": "Fantasy"}

        result = world_builder.build_world(
            manuscript_id,
            target_audience=target_audience,
            research_insights=research_insights,
        )

        assert isinstance(result, dict)
        assert "settings" in result
        mock_document_store.get_manuscript.assert_called_once_with(manuscript_id)
