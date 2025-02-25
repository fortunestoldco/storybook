from typing import List, Dict, Any
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.retrievers import TavilySearchAPIRetriever, WikipediaRetriever
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

class DocumentRetrieverTool(BaseTool):
    """Tool for retrieving documents from the vector store."""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        super().__init__(
            name="document_retriever",
            description="Retrieve relevant documents from the project's knowledge base",
            return_direct=True
        )

    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Run the tool to retrieve documents."""
        results = self.vector_store.similarity_search(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]

    async def _arun(self, query: str) -> List[Dict[str, Any]]:
        """Async version of document retrieval."""
        return await self._run(query)

class WebCrawlerTool(BaseTool):
    """Tool for crawling web pages and extracting information."""

    def __init__(self):
        super().__init__(
            name="web_crawler",
            description="Crawl web pages and extract relevant information",
            return_direct=True
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StoryBookBot/1.0; +http://example.com/bot)"
        }

    def _run(self, url: str) -> Dict[str, Any]:
        """Run the tool to crawl a web page."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract main content
            content = {
                "title": soup.title.string if soup.title else "",
                "text": self._extract_main_content(soup),
                "metadata": {
                    "url": url,
                    "timestamp": response.headers.get("date")
                }
            }

            return content
        except Exception as e:
            return {"error": str(e)}

    async def _arun(self, url: str) -> Dict[str, Any]:
        """Async version of web crawling."""
        return await self._run(url)

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from a web page."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if main_content:
            return " ".join(main_content.stripped_strings)
        return ""

class ToolsService:
    """Service for managing and providing tools to agents."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = MongoDBAtlas(
            embedding_function=self.embeddings,
            persist_directory="./data/vector_store"
        )

    def get_document_retriever(self) -> DocumentRetrieverTool:
        """Get the document retriever tool."""
        return DocumentRetrieverTool(self.vector_store)

    def get_web_crawler(self) -> WebCrawlerTool:
        """Get the web crawler tool."""
        return WebCrawlerTool()

    def get_research_tools(self) -> List[BaseTool]:
        """Get all research-related tools."""
        google_search = GoogleSearchAPIWrapper()
        google_books = Tool(
            name="google_books",
            description="Search for books using Google Books API",
            func=GoogleSearchAPIWrapper().search
        )
        tavily_search = TavilySearchAPIRetriever()
        wikipedia_retriever = WikipediaRetriever()

        return [
            self.get_document_retriever(),
            self.get_web_crawler(),
            create_retriever_tool(
                google_search,
                "google_search",
                "Search the web for information using Google"
            ),
            google_books,
            tavily_search,
            wikipedia_retriever
        ]

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store a document in the vector store."""
        ids = self.vector_store.add_texts(
            texts=[content],
            metadatas=[metadata]
        )
        return ids[0]
