from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from storybook.agents.base import BaseAgent
from storybook.config import create_llm, get_llm, TAVILY_API_KEY
from storybook.db.document_store import DocumentStore
from storybook.tools.research_tools import ResearchTools

logger = logging.getLogger(__name__)

class MarketResearcher(BaseAgent):  # Add inheritance
    """Agent responsible for market research and audience analysis."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)  # Add super call
        self.llm = get_llm(temperature=0.7)
        self.document_store = DocumentStore()
        self.search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

    def get_tools(self):
        """Get tools available to this agent."""
        return [
            self.document_tools.get_manuscript_tool(),
            self.document_tools.get_manuscript_search_tool(),
            self.research_tools.get_research_tool(),
            self.research_tools.get_web_crawl_tool(),
        ]

    def infer_genre_and_audience(self, manuscript_id: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Infer the genre and target audience of the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")
            return {}

        # Extract a representative sample
        content = manuscript.get("content", "")
        title = manuscript.get("title", "Untitled")

        # Take beginning, middle samples for analysis
        sample_size = min(5000, len(content) // 3)
        beginning = content[:sample_size]
        middle_start = max(0, (len(content) // 2) - (sample_size // 2))
        middle = content[middle_start : middle_start + sample_size]

        # Combine samples
        sample = f"Title: {title}\n\nBEGINNING SAMPLE:\n{beginning}\n\nMIDDLE SAMPLE:\n{middle}"

        # Define prompt for genre/audience inference
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Publishing Market Analyst specializing in identifying the genre, themes, and target audience of manuscripts.
        
        Based on the following manuscript sample, determine:
        
        1. The primary genre and any subgenres
        2. The likely target demographic (age range, gender if specifically targeted, interests)
        3. The key themes and topics addressed
        4. Similar published books this manuscript reminds you of
        5. The potential market positioning
        
        Manuscript Sample:
        {manuscript_sample}
        
        Format your response as a detailed analysis with specific sections for each element.
        Be as precise as possible about the target demographic, as this will guide market research.
        """
        )

        # Create the chain
        chain = (
            {"manuscript_sample": lambda _: sample}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis = chain.invoke("Analyze genre and audience")

        # Parse the analysis into sections
        sections = [
            "primary genre",
            "target demographic",
            "key themes",
            "similar books",
            "market positioning",
        ]

        result = {"full_analysis": analysis}

        # Extract sections
        for section in sections:
            pattern = re.compile(
                f"{section}s?:?(.+?)(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL
            )
            match = pattern.search(analysis)
            if match:
                key = section.lower().replace(" ", "_")
                result[key] = match.group(1).strip()

        return result

    def research_similar_books(self, genre: str, themes: List[str]) -> Dict[str, Any]:
        """Research similar books in the market based on genre and themes."""
        # Construct research queries
        queries = [
            f"bestselling {genre} novels published in the last 5 years",
            f"popular {genre} books with themes of {', '.join(themes[:3])}",
            f"{genre} book sales trends",
            f"reader demographics for {genre} fiction",
        ]

        research_results = []

        # Execute each research query
        for query in queries:
            research_tool = self.research_tools.get_research_tool()
            result = research_tool.invoke(query)

            # Add to research results
            research_results.append({"query": query, "results": result})

            # For each result, try to crawl one main URL for more in-depth information
            crawler = self.research_tools.get_web_crawl_tool()

            # Extract URLs from the result
            urls = re.findall(r"https?://\S+", result)
            if urls:
                # Limit to first URL to avoid excessive crawling
                try:
                    crawl_result = crawler.invoke(urls[0])
                    research_results.append(
                        {"query": f"Crawling {urls[0]}", "results": crawl_result}
                    )
                except Exception as e:
                    logger.error(f"Error crawling URL: {e}")

        # Define prompt to analyze research findings
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Publishing Market Analyst. Based on the following research findings about similar books 
        and market trends, create a comprehensive analysis of the market landscape.
        
        Research findings:
        {research_findings}
        
        Your analysis should cover:
        1. Top performing similar titles
        2. Common features of successful books in this category
        3. Current market trends
        4. Gaps or opportunities in the market
        5. Reader preferences and expectations
        
        Provide specific data points where available.
        """
        )

        # Format research findings
        findings_text = "\n\n".join(
            [f"Query: {r['query']}\nResults: {r['results']}" for r in research_results]
        )

        # Create the chain
        chain = (
            {"research_findings": lambda _: findings_text}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        market_analysis = chain.invoke("Analyze market")

        # Add the market analysis to the research results
        return {"raw_research": research_results, "market_analysis": market_analysis}

    def analyze_target_demographic(
        self, demographic_info: str, genre: str
    ) -> Dict[str, Any]:
        """Analyze the target demographic in depth."""
        # Format the research query for the target demographic
        query = f"reading preferences and buying habits of {demographic_info} in {genre} fiction"

        # Research the demographic
        research_tool = self.research_tools.get_research_tool()
        demographic_research = research_tool.invoke(query)

        # Additional queries to understand the demographic better
        additional_queries = [
            f"entertainment preferences of {demographic_info}",
            f"social media usage among {demographic_info}",
            f"cultural trends popular with {demographic_info}",
        ]

        additional_research = []
        for query in additional_queries:
            result = research_tool.invoke(query)
            additional_research.append({"query": query, "results": result})

            # For key insights, crawl relevant URLs for more detailed information
            urls = re.findall(r"https?://\S+", result)
            if urls:
                crawler = self.research_tools.get_web_crawl_tool()
                try:
                    # Limit to first URL to avoid excessive crawling
                    crawl_result = crawler.invoke(urls[0])
                    additional_research.append(
                        {
                            "query": f"Detailed analysis from {urls[0]}",
                            "results": crawl_result,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error crawling URL: {e}")

        # Define prompt to analyze demographic research
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Consumer Insights Specialist. Based on the following research about a target demographic,
        create a comprehensive profile that would help an author connect with this audience.
        
        Target Demographic: {demographic_info}
        Genre: {genre}
        
        Primary Research Findings:
        {demographic_research}
        
        Additional Insights:
        {additional_research}
        
        Your demographic profile should include:
        1. Reading preferences (formats, length, pacing, content preferences)
        2. Buying habits (where they discover books, purchase channels)
        3. Content sensitivities and preferences
        4. Values and interests that could be reflected in the book
        5. Language style and complexity expectations
        6. Character traits they typically connect with
        7. Specific recommendations for appealing to this demographic
        
        Be specific and actionable in your analysis.
        """
        )

        # Format additional research
        additional_text = "\n\n".join(
            [
                f"Query: {r['query']}\nResults: {r['results']}"
                for r in additional_research
            ]
        )

        # Create the chain
        chain = (
            {
                "demographic_info": lambda _: demographic_info,
                "genre": lambda _: genre,
                "demographic_research": lambda _: demographic_research,
                "additional_research": lambda _: additional_text,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        demographic_profile = chain.invoke("Create demographic profile")

        return {
            "demographic_info": demographic_info,
            "genre": genre,
            "raw_research": {
                "primary": demographic_research,
                "additional": additional_research,
            },
            "demographic_profile": demographic_profile,
        }

    def research_market(self, manuscript_id: str, title: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Step 1: Infer genre and target audience
            genre_audience = self.infer_genre_and_audience(manuscript_id, llm_config)

            if not genre_audience:
                return {
                    "manuscript_id": manuscript_id,
                    "status": "inference_failed",
                    "message": "Failed to infer genre and target audience.",
                }

            # Extract key information
            genre = genre_audience.get("primary_genre", "fiction")
            demographic = genre_audience.get("target_demographic", "general readers")
            themes = []

            # Extract themes
            themes_text = genre_audience.get("key_themes", "")
            theme_matches = re.findall(
                r'["\']([^"\']+)["\']|(\w+(?:\s+\w+){0,2})', themes_text
            )
            for match in theme_matches:
                theme = match[0] if match[0] else match[1]
                if theme and len(theme) > 3 and theme.lower() not in ["and", "the", "with"]:
                    themes.append(theme)

            if not themes:
                themes = ["adventure", "relationships", "personal growth"]  # Default themes

            # Step 2: Research similar books
            market_research = self.research_similar_books(genre, themes)

            # Step 3: Analyze target demographic
            demographic_analysis = self.analyze_target_demographic(demographic, genre)

            # Step 4: Generate comprehensive insights
            prompt = ChatPromptTemplate.from_template(
                """
            You are a Publishing Consultant preparing a comprehensive market research report for an author.
            
            Manuscript Title: {title}
            Genre: {genre}
            Target Demographic: {demographic}
            Key Themes: {themes}
            
            Market Analysis:
            {market_analysis}
            
            Demographic Profile:
            {demographic_profile}
            
            Based on all this research, prepare a comprehensive report covering:
            
            1. Market Overview: Current state of the {genre} market
            2. Competition: How similar books are performing and positioning
            3. Target Audience: Detailed profile of the ideal reader
            4. Market Opportunities: Gaps or trends the author could leverage
            5. Positioning Recommendations: How to position this book for success
            6. Content Recommendations: Specific elements to include or emphasize to appeal to the target audience
            7. Marketing Considerations: Channels and approaches likely to reach this audience
            
            Your report should provide actionable insights that will guide decisions about character development,
            dialogue style, world-building, and language choices to maximize appeal to the target audience.
            """
            )

            # Create the chain
            chain = (
                {
                    "title": lambda _: title,
                    "genre": lambda _: genre,
                    "demographic": lambda _: demographic,
                    "themes": lambda _: ", ".join(themes),
                    "market_analysis": lambda _: market_research.get("market_analysis", ""),
                    "demographic_profile": lambda _: demographic_analysis.get(
                        "demographic_profile", ""
                    ),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            comprehensive_report = chain.invoke("Generate market report")

            # Store the research in the database
            research_data = {
                "title": f"Market Research Report for '{title}'",
                "genre_analysis": genre_audience,
                "market_research": market_research,
                "demographic_analysis": demographic_analysis,
                "comprehensive_report": comprehensive_report,
            }

            self.document_store.store_research_document(
                manuscript_id, "market_research", research_data
            )

            # Store in MongoDB Atlas Vector store for future reference and search
            doc = Document(
                page_content=comprehensive_report,
                metadata={
                    "type": "market_research",
                    "manuscript_id": manuscript_id,
                    "title": title,
                    "genre": genre,
                    "demographic": demographic,
                    "themes": ", ".join(themes),
                },
            )

            self.document_store.db.store_documents_with_embeddings("research", [doc])

            # Return the complete research package
            return {
                "manuscript_id": manuscript_id,
                "status": "success",
                "message": "Completed comprehensive market research.",
                "research_insights": {
                    "genre": genre,
                    "target_demographic": demographic,
                    "key_themes": themes,
                    "market_analysis_summary": market_research.get("market_analysis", "")[
                        :500
                    ]
                    + "...",
                    "comprehensive_report": comprehensive_report,
                },
                "target_audience": {
                    "demographic": demographic,
                    "profile": demographic_analysis.get("demographic_profile", ""),
                    "reading_preferences": self._extract_preferences(
                        demographic_analysis.get("demographic_profile", "")
                    ),
                    "content_expectations": self._extract_expectations(
                        demographic_analysis.get("demographic_profile", "")
                    ),
                    "genre": genre,
                },
            }
        except Exception as e:
            logger.error(f"Error in market research: {e}")
            return self.handle_error(e)

    def _extract_preferences(self, profile_text: str) -> Dict[str, Any]:
        """Extract reading preferences from the demographic profile."""
        preferences = {}

        # Look for reading preferences
        reading_pref_pattern = re.compile(
            r"reading preferences:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = reading_pref_pattern.search(profile_text)
        if match:
            preferences["reading"] = match.group(1).strip()

        # Look for language style preferences
        language_pattern = re.compile(
            r"language style:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = language_pattern.search(profile_text)
        if match:
            preferences["language"] = match.group(1).strip()

        # Look for character preferences
        character_pattern = re.compile(
            r"character traits:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = character_pattern.search(profile_text)
        if match:
            preferences["characters"] = match.group(1).strip()

        return preferences

    def _extract_expectations(self, profile_text: str) -> Dict[str, Any]:
        """Extract content expectations from the demographic profile."""
        expectations = {}

        # Look for content sensitivities
        sensitivity_pattern = re.compile(
            r"content sensitivities:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = sensitivity_pattern.search(profile_text)
        if match:
            expectations["sensitivities"] = match.group(1).strip()

        # Look for values and interests
        values_pattern = re.compile(
            r"values and interests:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = values_pattern.search(profile_text)
        if match:
            expectations["values"] = match.group(1).strip()

        # Look for recommendations
        recommendations_pattern = re.compile(
            r"recommendations:?(.+?)(?=\n\n|\n\d\.|\Z)", re.IGNORECASE | re.DOTALL
        )
        match = recommendations_pattern.search(profile_text)
        if match:
            expectations["recommendations"] = match.group(1).strip()

        return expectations
