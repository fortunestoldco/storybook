from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional
import logging
import json
import re
from datetime import datetime

# Third-party imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Local imports
from storybook.agents.base import BaseAgent
from storybook.config import create_llm, get_llm, TAVILY_API_KEY
from storybook.db.document_store import DocumentStore
from storybook.tools.research_tools import ResearchTools

logger = logging.getLogger(__name__)

class MarketResearcher(BaseAgent):
    """Agent responsible for market research and audience analysis."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.search_tool = TavilySearchAPIWrapper(api_key=TAVILY_API_KEY)
        self.research_tools = ResearchTools()

    def research_market(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conduct market research for the manuscript."""
        try:
            if not self.validate_input(manuscript_id=manuscript_id):
                return {"error": "Invalid manuscript_id"}
            if llm_config:
                self.llm = create_llm(llm_config)

            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # Infer genre and target audience
            genre_info = self._infer_genre_and_audience(manuscript["content"])

            # Research similar books and market trends
            market_research = self._research_similar_books(
                genre_info["genre"],
                genre_info["themes"],
                genre_info["target_demographic"]
            )

            # Analyze target demographic
            demographic_analysis = self._analyze_target_demographic(
                genre_info["target_demographic"],
                market_research
            )

            # Generate comprehensive insights
            insights = self._generate_market_insights(
                manuscript_id,
                genre_info,
                market_research,
                demographic_analysis
            )

            return {
                "manuscript_id": manuscript_id,
                "genre_analysis": genre_info,
                "market_research": market_research,
                "demographic_analysis": demographic_analysis,
                "insights": insights
            }

        except Exception as e:
            logger.error(f"Error in research_market: {str(e)}")
            return self.handle_error(e)

    def process_manuscript(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]], research_insights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process manuscript for market research."""
        try:
            return self.research_market(
                manuscript_id,
                target_audience,
                research_insights
            )
        except Exception as e:
            logger.error(f"Error in market research: {str(e)}")
            return self.handle_error(e)

    def _infer_genre_and_audience(self, content: str) -> Dict[str, Any]:
        """Infer the genre and target audience of the manuscript."""
        try:
            # Take representative samples from beginning, middle, and end
            sample_size = min(2000, len(content) // 3)
            sample = f"{content[:sample_size]}\n...\n{content[-sample_size:]}"

            prompt = ChatPromptTemplate.from_template(
                """Analyze this manuscript sample to determine:
                1. Primary genre and subgenres
                2. Major themes and motifs
                3. Target audience characteristics
                4. Similar published works

                Manuscript Sample:
                {manuscript_sample}

                Format your response as a structured analysis with clear sections.
                Be as specific as possible about the target demographic.
                """
            )

            chain = (
                {"manuscript_sample": lambda _: sample}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            analysis = chain.invoke("Analyze genre and audience")

            # Extract key information using regex
            genre_match = re.search(r"genre:?\s*(.*?)(?=\n|$)", analysis, re.IGNORECASE)
            genre = genre_match.group(1).strip() if genre_match else "fiction"

            themes_text = re.search(r"themes:?\s*(.*?)(?=\n\n|\Z)", analysis, re.IGNORECASE)
            themes = []
            if themes_text:
                themes = [t.strip() for t in themes_text.group(1).split(",")]

            demographic_match = re.search(
                r"target audience:?\s*(.*?)(?=\n\n|\Z)",
                analysis,
                re.IGNORECASE
            )
            demographic = (
                demographic_match.group(1).strip()
                if demographic_match
                else "general readers"
            )

            return {
                "genre": genre,
                "themes": themes[:3],  # Top 3 themes
                "target_demographic": demographic,
                "full_analysis": analysis
            }

        except Exception as e:
            logger.error(f"Error inferring genre and audience: {str(e)}")
            return {
                "genre": "fiction",
                "themes": [],
                "target_demographic": "general readers",
                "full_analysis": ""
            }

    def _research_similar_books(
        self,
        genre: str,
        themes: List[str],
        demographic: str
    ) -> Dict[str, Any]:
        """Research similar books and market trends."""
        try:
            queries = [
                f"bestselling {genre} books with themes of {', '.join(themes[:3])}",
                f"reader demographics for {genre} fiction",
                f"market trends in {genre} publishing",
                f"successful {genre} books for {demographic}"
            ]

            research_results = []
            for query in queries:
                result = self.search_tool.run(query)
                research_results.append({"query": query, "results": result})

            # Extract URLs for deeper analysis
            urls = []
            for result in research_results:
                urls.extend(re.findall(r"https?://\S+", result["results"]))

            # Crawl one main URL for more information if available
            additional_research = []
            if urls and self.research_tools:  # Add null check
                try:
                    crawl_tool = self.research_tools.get_web_crawl_tool()
                    if crawl_tool:  # Add null check
                        crawl_result = crawl_tool.run(urls[0])
                        additional_research.append({
                            "url": urls[0],
                            "content": crawl_result
                        })
                except Exception as e:
                    logger.warning(f"Error crawling URL: {str(e)}")

            return {
                "raw_research": research_results,
                "additional_research": additional_research
            }

        except Exception as e:
            logger.error(f"Error researching similar books: {str(e)}")
            return {"raw_research": [], "additional_research": []}

    def _analyze_target_demographic(
        self,
        demographic: str,
        market_research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the target demographic in detail."""
        try:
            additional_queries = [
                f"reading preferences of {demographic}",
                f"book buying habits of {demographic}",
                f"social media usage among {demographic}"
            ]

            demographic_research = []
            for query in additional_queries:
                try:
                    result = self.search_tool.run(query, timeout=10)  # Add timeout
                    demographic_research.append({"query": query, "results": result})
                except TimeoutError:
                    logger.warning(f"Search timeout for query: {query}")
                    continue

            return {
                "demographic": demographic,
                "research": demographic_research,
                "preferences": self._extract_preferences(demographic_research),
                "recommendations": self._extract_recommendations(demographic_research)
            }

        except Exception as e:
            logger.error(f"Error analyzing target demographic: {str(e)}")
            return {"demographic": demographic, "research": []}

    def _generate_market_insights(
        self,
        manuscript_id: str,
        genre_info: Dict[str, Any],
        market_research: Dict[str, Any],
        demographic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive market insights."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """As a Market Research Specialist, analyze this data and provide:
                1. Market Overview: Current state and trends
                2. Target Audience: Detailed reader profile
                3. Competition Analysis: Similar books and their performance
                4. Market Opportunities: Gaps and potential
                5. Marketing Recommendations: How to reach the audience

                Genre Information:
                {genre_info}

                Market Research:
                {market_research}

                Demographic Analysis:
                {demographic_analysis}

                Provide a comprehensive report with actionable insights.
                """
            )

            chain = (
                {
                    "genre_info": lambda _: json.dumps(genre_info, indent=2),
                    "market_research": lambda _: json.dumps(market_research.get("raw_research", []), indent=2),
                    "demographic_analysis": lambda _: json.dumps(demographic_analysis, indent=2)
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            insights = chain.invoke("Generate market insights")

            # Store the insights for future reference
            doc = Document(
                page_content=insights,
                metadata={
                    "type": "market_research",
                    "manuscript_id": manuscript_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "genre": genre_info.get("genre", "unknown"),
                    "demographic": demographic_analysis.get("demographic", "unknown")
                }
            )
            self.document_store.store_documents_with_embeddings("research", [doc])

            return {
                "comprehensive_report": insights,
                "key_recommendations": self._extract_recommendations(insights)
            }

        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            return {"comprehensive_report": "", "key_recommendations": []}

    def _extract_preferences(self, research: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract reading preferences from research data."""
        preferences = {
            "format": [],
            "genre": [],
            "content": [],
            "pricing": []
        }
        
        try:
            for item in research:
                text = item["results"].lower()
                
                # Extract format preferences using raw string
                formats = re.findall(fr"prefer\s+(ebook|audiobook|print|hardcover|paperback)", text)
                preferences["format"].extend(formats)
                
                # Extract genre preferences
                genres = re.findall(fr"prefer\s+(\w+)\s+(?:books|fiction|novels)", text)
                preferences["genre"].extend(genres)
                
                # Extract content preferences
                content = re.findall(fr"prefer\s+([\w\s]+)\s+content", text)
                preferences["content"].extend(content)
                
                # Extract pricing preferences
                if "price" in text or "pricing" in text:
                    pricing = re.findall(r"\$\d+(?:\.\d{2})?", text)
                    preferences["pricing"].extend(pricing)
            
            # Remove duplicates and sort
            for key in preferences:
                preferences[key] = sorted(list(set(preferences[key])))
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error extracting preferences: {str(e)}")
            return preferences

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract marketing recommendations from text."""
        try:
            recommendations = []
            recommendation_pattern = re.compile(
                r"(?:recommend|suggest|advise)\s*(.*?)(?=\.|$)",
                re.IGNORECASE
            )
            
            matches = recommendation_pattern.findall(text)
            recommendations.extend([m.strip() for m in matches if m.strip()])
            
            return list(set(recommendations))
            
        except Exception as e:
            logger.error(f"Error extracting recommendations: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        try:
            # Remove special characters
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            # Normalize whitespace
            text = ' '.join(text.split())
            return text.strip().lower()
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text