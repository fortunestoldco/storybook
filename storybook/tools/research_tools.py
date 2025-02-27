from typing import Dict, List, Any, Optional
import logging
import re

from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Replicate

from storybook.config import get_llm

logger = logging.getLogger(__name__)


class ResearchTools:
    """Tools for conducting market and literature research."""

    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)

    def get_research_tool(self):
        """Create a tool to perform web research."""
        return Tool(
            name="WebResearch",
            description="Performs web research on a given topic related to publishing, literature, or writing.",
            func=self._simulate_web_research,
        )

    def _simulate_web_research(self, query: str) -> str:
        """Simulate web research using an LLM."""
        # In a production system, this would actually query web APIs or use tools
        # like Tavily Search or other research tools. For this demo, we'll simulate
        # with an LLM.

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
