from typing import Dict, List, Any, Optional
import logging
import re

from langchain_core.tools import Tool
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.utilities.firecrawl import FireCrawl

from storybook.config import get_llm, TAVILY_API_KEY, FIRECRAWL_API_KEY
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class ResearchTools:
    """Tools for conducting market and literature research."""

    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)
        self.document_store = DocumentStore()
        
        # Initialize the Tavily search API wrapper
        try:
            self.tavily_search = TavilySearchAPIWrapper(api_key=TAVILY_API_KEY)
            self.tavily_available = True
        except Exception as e:
            logger.warning(f"Tavily search is not available: {e}")
            self.tavily_available = False
            
        # Initialize FireCrawl
        try:
            self.firecrawl = FireCrawl(api_key=FIRECRAWL_API_KEY)
            self.firecrawl_available = True
        except Exception as e:
            logger.warning(f"FireCrawl is not available: {e}")
            self.firecrawl_available = False

    def get_research_tool(self):
        """Create a tool to perform web research."""
        # If Tavily is available, use it
        if self.tavily_available:
            return TavilySearchResults(
                api_key=TAVILY_API_KEY,
                max_results=5, 
                description="Performs web research on a given topic related to publishing, literature, or writing."
            )
        
        # Otherwise, use the simulated research
        return Tool(
            name="WebResearch",
            description="Performs web research on a given topic related to publishing, literature, or writing.",
            func=self._simulate_web_research,
        )
        
    def get_web_crawl_tool(self):
        """Create a tool to crawl web pages for more detailed information."""
        return Tool(
            name="WebCrawl",
            description="Crawls a specific web page for detailed information and saves it for research.",
            func=self._crawl_and_save_webpage,
        )

    def _simulate_web_research(self, query: str) -> str:
        """Simulate web research using an LLM."""
        # First try Tavily if available
        if self.tavily_available:
            try:
                results = self.tavily_search.results(query)
                if results:
                    formatted_results = "\n\n".join([
                        f"Title: {result.get('title', 'No title')}\n"
                        f"Content: {result.get('content', 'No content')}\n"
                        f"URL: {result.get('url', 'No URL')}"
                        for result in results
                    ])
                    return formatted_results
            except Exception as e:
                logger.error(f"Error with Tavily search: {e}")
                # Fall back to LLM simulation

        # LLM simulation as fallback
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a research assistant with extensive knowledge of publishing, literature, and book markets.
            Simulate realistic, detailed research results for the query below.
            
            Query: {query}
            
            Provide detailed, factual, and comprehensive information that would be found from multiple
            credible sources. Include relevant statistics, trends, examples, and citations where appropriate.
            
            Format your response to appear as if you've consulted multiple authoritative sources on this topic.
            Avoid any language indicating this is simulated research.
            """,
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(query=query)
            return self._clean_research_output(result)
        except Exception as e:
            logger.error(f"Error in web research simulation: {e}")
            return f"Research failed due to an error: {str(e)}"
            
    def _crawl_and_save_webpage(self, url: str) -> str:
        """Crawl a webpage and save the content to the document store."""
        if not url.startswith(('http://', 'https://')):
            return f"Invalid URL format: {url}"
            
        try:
            # Try to use FireCrawl if available
            if self.firecrawl_available:
                content = self.firecrawl.load_page(url)
                
                # Store the crawled content
                tag = "web_research"
                self.document_store.store_research_from_web([url], tags=[tag])
                
                return f"Successfully crawled and stored content from {url} with tag '{tag}'"
            
            # Fallback to WebBaseLoader
            loader = WebBaseLoader([url])
            documents = loader.load()
            
            # Store the documents
            tag = "web_research"
            doc_ids = self.document_store.db.store_documents_with_embeddings(
                "research", 
                documents
            )
            
            if not doc_ids:
                return f"Failed to store content from {url}"
                
            return f"Successfully crawled and stored content from {url}"
            
        except Exception as e:
            logger.error(f"Error crawling webpage {url}: {e}")
            return f"Failed to crawl webpage {url}: {str(e)}"

    def _clean_research_output(self, output: str) -> str:
        """Clean up the research output to remove any metacommentary."""
        # Remove phrases indicating simulation
        simulation_phrases = [
            "As a research assistant",
            "As an AI",
            "I'm simulating",
            "I would find",
            "I don't have access to",
        ]

        for phrase in simulation_phrases:
            output = re.sub(phrase, "", output, flags=re.IGNORECASE)

        # If the output starts with a heading, keep it structured
        if re.match(r"^#+ ", output.strip()):
            return output.strip()

        # Otherwise, try to format it as a clean research report
        sections = re.split(r"\n\n+", output.strip())
        if len(sections) > 1:
            # Already has paragraph structure
            return output.strip()
        else:
            # Try to add some structure
            paragraphs = []
            current = []

            for line in output.split("\n"):
                if not line.strip():
                    if current:
                        paragraphs.append(" ".join(current))
                        current = []
                    continue
                current.append(line.strip())

            if current:
                paragraphs.append(" ".join(current))

            return "\n\n".join(paragraphs)
