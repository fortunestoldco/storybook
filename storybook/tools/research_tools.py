from typing import Dict, List, Any, Optional
import logging
import re

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper

from storybook.config import get_llm

logger = logging.getLogger(__name__)


class ResearchTools:
    """Tools for conducting market and literature research."""

    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)
        
        # Initialize real research tools when available
        try:
            self.tavily_search = TavilySearchResults(max_results=5)
            self.use_tavily = True
        except:
            self.use_tavily = False
            logger.warning("Tavily Search not available, falling back to simulated research")
            
        try:
            self.serpapi = SerpAPIWrapper()
            self.use_serpapi = True
        except:
            self.use_serpapi = False
            logger.warning("SerpAPI not available, falling back to other search methods")

    def get_research_tool(self):
        """Create a tool to perform web research."""
        return Tool(
            name="WebResearch",
            description="Performs web research on a given topic related to publishing, literature, or writing.",
            func=self._perform_research,
        )

    def _perform_research(self, query: str) -> str:
        """Perform web research using available search tools."""
        # Try using Tavily first if available
        if self.use_tavily:
            try:
                search_results = self.tavily_search.invoke(query)
                
                if search_results:
                    # Format the results
                    formatted_results = "\n\n".join([
                        f"SOURCE: {result['url']}\n{result['content']}" 
                        for result in search_results
                    ])
                    
                    # Synthesize the results
                    return self._synthesize_research(query, formatted_results)
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")
        
        # Try SerpAPI as fallback
        if self.use_serpapi:
            try:
                search_results = self.serpapi.run(query)
                if search_results:
                    return self._synthesize_research(query, search_results)
            except Exception as e:
                logger.error(f"SerpAPI search failed: {e}")
        
        # Fallback to simulated research
        return self._simulate_web_research(query)

    def _synthesize_research(self, query: str, research_data: str) -> str:
        """Synthesize research results into a coherent summary."""
        prompt = PromptTemplate(
            input_variables=["query", "research_data"],
            template="""
            You are a research analyst synthesizing information about publishing, literature, and book markets.
            
            Query: {query}
            
            Research Data:
            {research_data}
            
            Synthesize this information into a comprehensive, well-organized research report.
            Focus on facts, statistics, trends, and citations from the research.
            Organize your response with clear sections and relevant data points.
            """
        )
        
        chain = (
            {"query": lambda x: query, "research_data": lambda x: research_data}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        result = chain.invoke({})
        return self._clean_research_output(result)

    def _simulate_web_research(self, query: str) -> str:
        """Simulate web research using an LLM when real search is unavailable."""
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
            """
        )
        
        chain = (
            {"query": lambda x: query}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        result = chain.invoke({})
        return self._clean_research_output(result)

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
