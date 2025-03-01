import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.agents.characterdev import CharacterDeveloper


class TestCharacterDeveloper:
    @pytest.fixture
    def character_developer(self, mock_llm, mock_document_store):
        return CharacterDeveloper()

    def test_enhance_characters(self, character_developer, mock_document_store):
        manuscript_id = "test_123"
        target_audience = {"demographic": "Young Adult"}
        research_insights = {"genre": "Fantasy"}

        result = character_developer.enhance_characters(
            manuscript_id,
            target_audience=target_audience,
            research_insights=research_insights,
        )

        assert isinstance(result, dict)
        assert "characters" in result
        mock_document_store.get_manuscript.assert_called_once_with(manuscript_id)
