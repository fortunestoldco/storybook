# test_*.py files need pytest and MagicMock imports
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.tools.research_tools import ResearchTools
from langchain_core.tools import Tool
from langchain_community.document_loaders import FireCrawlLoader


class TestResearchTools:

    @pytest.fixture
    def mock_llm(self):
        with patch("storybook.tools.research_tools.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            yield mock_llm

    @pytest.fixture
    def mock_tavily_search(self):
        with patch(
            "storybook.tools.research_tools.TavilySearchAPIWrapper"
        ) as mock_tavily:
            mock_search = MagicMock()
            mock_tavily.return_value = mock_search
            yield mock_search

    @pytest.fixture
    def mock_tavily_tool(self):
        with patch(
            "storybook.tools.research_tools.TavilySearchResults"
        ) as mock_tavily_tool:
            mock_tool = MagicMock()
            mock_tavily_tool.return_value = mock_tool
            yield mock_tool

    @pytest.fixture
    def mock_firecrawl(self):
        with patch(
            "langchain_community.document_loaders.FireCrawlLoader"
        ) as mock_firecrawl:
            mock_crawl = MagicMock()
            mock_firecrawl.return_value = mock_crawl
            yield mock_crawl

    @pytest.fixture
    def mock_document_store(self):
        with patch("storybook.tools.research_tools.DocumentStore") as mock_doc_store:
            mock_store = MagicMock()
            mock_doc_store.return_value = mock_store
            yield mock_store

    def test_init(
        self, mock_llm, mock_tavily_search, mock_firecrawl, mock_document_store
    ):
        # Test initialization
        tools = ResearchTools()

        # Verify LLM was initialized
        assert tools.llm == mock_llm

        # Verify Tavily search was initialized
        assert tools.tavily_search == mock_tavily_search
        assert tools.tavily_available == True

        # Verify FireCrawl was initialized
        assert tools.firecrawl == mock_firecrawl
        assert tools.firecrawl_available == True

        # Verify document store was initialized
        assert tools.document_store == mock_document_store

        # Test initialization with Tavily error
        mock_tavily_search.side_effect = Exception("Tavily error")
        tools = ResearchTools()
        assert tools.tavily_available == False

        # Reset mock for further tests
        mock_tavily_search.side_effect = None

        # Test initialization with FireCrawl error
        mock_firecrawl.side_effect = Exception("FireCrawl error")
        tools = ResearchTools()
        assert tools.firecrawl_available == False

    def test_get_research_tool(
        self, mock_llm, mock_tavily_search, mock_tavily_tool, mock_document_store
    ):
        # Test getting research tool with Tavily available
        tools = ResearchTools()
        tool = tools.get_research_tool()

        # Verify Tavily tool was returned
        assert tool == mock_tavily_tool

        # Test getting research tool with Tavily unavailable
        tools.tavily_available = False
        tool = tools.get_research_tool()

        # Verify a fallback tool was returned
        assert isinstance(tool, Tool)
        assert tool.name == "WebResearch"

    def test_get_web_crawl_tool(
        self, mock_llm, mock_tavily_search, mock_firecrawl, mock_document_store
    ):
        # Test getting web crawl tool
        tools = ResearchTools()
        tool = tools.get_web_crawl_tool()

        # Verify a Tool was returned
        assert isinstance(tool, Tool)
        assert tool.name == "WebCrawl"

    def test_simulate_web_research(
        self, mock_llm, mock_tavily_search, mock_document_store
    ):
        # Setup
        tools = ResearchTools()

        # Mock Tavily search results
        mock_tavily_search.results.return_value = [
            {"title": "Test Title", "content": "Test Content", "url": "http://test.com"}
        ]

        # Test with Tavily available
        result = tools._simulate_web_research("test query")

        # Verify Tavily was used
        mock_tavily_search.results.assert_called_once_with("test query")
        assert "Test Title" in result
        assert "Test Content" in result
        assert "http://test.com" in result

        # Test with Tavily error
        mock_tavily_search.results.side_effect = Exception("Tavily error")

        # Mock LLM chain
        mock_chain = MagicMock()
        mock_chain.run.return_value = "Simulated research result"

        with patch("storybook.tools.research_tools.LLMChain") as mock_llm_chain:
            mock_llm_chain.return_value = mock_chain

            result = tools._simulate_web_research("test query")

            # Verify LLM was used as fallback
            mock_llm_chain.assert_called_once()
            mock_chain.run.assert_called_once_with(query="test query")
            assert result == "Simulated research result"

    def test_crawl_and_save_webpage(
        self, mock_llm, mock_firecrawl, mock_document_store
    ):
        # Setup
        tools = ResearchTools()

        # Test with invalid URL
        result = tools._crawl_and_save_webpage("invalid_url")
        assert "Invalid URL format" in result

        # Test with FireCrawl available
        mock_firecrawl.load_page.return_value = "Crawled content"
        result = tools._crawl_and_save_webpage("https://test.com")

        # Verify FireCrawl was used
        mock_firecrawl.load_page.assert_called_once_with("https://test.com")
        assert "Successfully crawled" in result

        # Test with FireCrawl unavailable
        tools.firecrawl_available = False

        # Mock WebBaseLoader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [MagicMock()]

        with patch("storybook.tools.research_tools.WebBaseLoader") as mock_web_loader:
            mock_web_loader.return_value = mock_loader

            # Mock document store to return doc IDs
            mock_document_store.db.store_documents_with_embeddings.return_value = [
                "doc1"
            ]

            result = tools._crawl_and_save_webpage("https://test.com")

            # Verify WebBaseLoader was used as fallback
            mock_web_loader.assert_called_once_with(["https://test.com"])
            mock_loader.load.assert_called_once()
            assert "Successfully crawled" in result

            # Test with storage failure
            mock_document_store.db.store_documents_with_embeddings.return_value = []

            result = tools._crawl_and_save_webpage("https://test.com")
            assert "Failed to store content" in result

    def test_clean_research_output(
        self, mock_llm, mock_tavily_search, mock_document_store
    ):
        # Setup
        tools = ResearchTools()

        # Test with simulation phrases
        output = "As a research assistant, I found that the market is growing. I'm simulating research results."
        cleaned = tools._clean_research_output(output)

        # Verify simulation phrases were removed
        assert "As a research assistant" not in cleaned
        assert "I'm simulating" not in cleaned
        assert "the market is growing" in cleaned

        # Test with heading structure
        output = "# Research Findings\n\nThe market is growing.\n\n## Market Trends\n\nTrends show increase."
        cleaned = tools._clean_research_output(output)

        # Verify structure was preserved
        assert "# Research Findings" in cleaned
        assert "## Market Trends" in cleaned

        # Test with paragraph structure
        output = "Research shows growth.\n\nTrends indicate expansion.\n\nExperts predict continued rise."
        cleaned = tools._clean_research_output(output)

        # Verify paragraphs were preserved
        assert "Research shows growth." in cleaned
        assert "Trends indicate expansion." in cleaned
        assert cleaned.count("\n\n") == 2
