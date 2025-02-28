from unittest.mock import patch, MagicMock
import pytest
from storybook.agents.content_analyzer import ContentAnalyzer
from storybook.config import LLMProvider

class TestContentAnalyzer:
    @pytest.fixture
    def content_analyzer(self):
        return ContentAnalyzer()

    @pytest.fixture
    def mock_document_store(self):
        with patch("storybook.db.document_store.DocumentStore") as mock:
            mock.return_value.get_manuscript.return_value = {
                "content": "Test manuscript content",
                "title": "Test Title"
            }
            yield mock

    def test_analyze_content_with_openai(self, mock_document_store, mock_llm):
        """Test content analysis with OpenAI configuration."""
        llm_config = {
            "provider": LLMProvider.OPENAI,
            "config": {
                "model_name": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 2000,
                "streaming": False
            }
        }
        
        analyzer = ContentAnalyzer(llm_config=llm_config)
        result = analyzer.analyze_content("test_123")
        
        assert isinstance(result, dict)
        assert "analysis" in result
        mock_llm.assert_called_once()

    def test_analyze_content_with_anthropic(self, mock_document_store, mock_llm):
        """Test content analysis with Anthropic configuration."""
        llm_config = {
            "provider": LLMProvider.ANTHROPIC,
            "config": {
                "model_name": "claude-3-opus",
                "temperature": 0.7,
                "max_tokens": 4000,
                "streaming": False
            }
        }
        
        analyzer = ContentAnalyzer(llm_config=llm_config)
        result = analyzer.analyze_content("test_123")
        
        assert isinstance(result, dict)
        assert "analysis" in result

    def test_runtime_llm_config_change(self, mock_document_store, mock_llm):
        """Test changing LLM configuration at runtime."""
        initial_config = {
            "provider": LLMProvider.OPENAI,
            "config": {
                "model_name": "gpt-4",
                "temperature": 0.5
            }
        }
        
        runtime_config = {
            "provider": LLMProvider.ANTHROPIC,
            "config": {
                "model_name": "claude-3-opus",
                "temperature": 0.7
            }
        }
        
        analyzer = ContentAnalyzer(llm_config=initial_config)
        
        # First analysis with initial config
        result1 = analyzer.analyze_content("test_123")
        assert isinstance(result1, dict)
        
        # Second analysis with runtime config
        result2 = analyzer.analyze_content("test_123", llm_config=runtime_config)
        assert isinstance(result2, dict)
        
        # Verify LLM was reconfigured
        assert mock_llm.call_count == 2

    def test_invalid_llm_config(self, mock_document_store):
        """Test handling of invalid LLM configuration."""
        invalid_config = {
            "provider": "invalid_provider",
            "config": {
                "model_name": "invalid_model"
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            ContentAnalyzer(llm_config=invalid_config)
        
        assert "Unsupported LLM provider" in str(exc_info.value)

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        with patch("storybook.config.create_llm") as mock:
            mock.return_value = MagicMock()
            yield mock