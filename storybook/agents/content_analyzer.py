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
from langchain_core.documents import Document
from langchain_mongodb.docstores import MongoDBDocStore

# Local imports
from storybook.config import create_llm, get_llm
from storybook.db.mongodb_client import MongoDBStore  # Fixed import path
from storybook.db.document_store import DocumentStore
from storybook.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ContentAnalyzer(BaseAgent):
    """Agent responsible for analyzing manuscript content."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize ContentAnalyzer with optional LLM configuration."""
        super().__init__(llm_config)  # Initialize base agent
        self.document_tools = DocumentTools()
        self.embeddings = OpenAIEmbeddings()
        self.db = MongoDBStore()

    def get_tools(self):
        """Get tools available to this agent."""
        return [
            self.document_tools.get_manuscript_tool(),
            self.document_tools.get_manuscript_search_tool(),
        ]

    def analyze_content(self, manuscript_id: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze manuscript content with optional runtime LLM configuration."""
        try:
            # Update LLM if new config provided
            if llm_config:
                self.llm = create_llm(llm_config)
                
            logger.info(f"Analyzing content for manuscript {manuscript_id}")
            
            # Get manuscript content from document store
            document = self.document_store.get_manuscript(manuscript_id)
            if not document:
                raise ValueError(f"Manuscript {manuscript_id} not found")
            
            content = document.get("content", "")
            
            # Perform comprehensive content analysis
            analysis = {
                "sentiment": self._analyze_sentiment(content),
                "readability": self._analyze_readability(content),
                "content_structure": self._analyze_structure(content),
                "genre_match": self._analyze_genre(content),
                "themes": self._identify_themes(content),
                "key_elements": {
                    "plot_points": self._extract_plot_points(content),
                    "characters": self._extract_characters(content),
                    "settings": self._extract_settings(content),
                }
            }
            
            return {
                "analysis": analysis,
                "message": "Content analysis completed successfully"
            }
        except Exception as e:
            logger.error(f"Error analyzing content for manuscript {manuscript_id}: {e}")
            return {"error": str(e)}

    def analyze_progress(self, manuscript_id: str, previous_analysis: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Analyze progress made after a specific stage of the transformation."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")
            return {}

        content = manuscript.get("content", "")

        # Define what to analyze based on the stage
        progress_analysis = {}

        if stage == "story_arc":
            # Analyze structure improvements
            new_structure = self._analyze_structure(content)
            progress_analysis["structure_improvements"] = self._compare_analysis(
                previous_analysis.get("content_structure", {}), new_structure
            )
            progress_analysis["content_structure"] = new_structure

        elif stage == "language":
            # Analyze style and readability improvements
            new_style = self._analyze_style(content)
            new_readability = self._analyze_readability(content)

            progress_analysis["style_improvements"] = self._compare_analysis(
                previous_analysis.get("writing_style", {}), new_style
            )
            progress_analysis["readability_improvements"] = self._compare_analysis(
                previous_analysis.get("readability", {}), new_readability
            )

            progress_analysis["writing_style"] = new_style
            progress_analysis["readability"] = new_readability

        elif stage == "complete":
            # Comprehensive final analysis
            new_sentiment = self._analyze_sentiment(content)
            new_readability = self._analyze_readability(content)
            new_structure = self._analyze_structure(content)
            new_style = self._analyze_style(content)

            # Compare all metrics with original
            progress_analysis["overall_improvements"] = {
                "sentiment": self._compare_analysis(
                    previous_analysis.get("sentiment", {}), new_sentiment
                ),
                "readability": self._compare_analysis(
                    previous_analysis.get("readability", {}), new_readability
                ),
                "structure": self._compare_analysis(
                    previous_analysis.get("content_structure", {}), new_structure
                ),
                "style": self._compare_analysis(
                    previous_analysis.get("writing_style", {}), new_style
                ),
            }

            # Generate improvement summary
            progress_analysis["improvement_summary"] = (
                self._generate_improvement_summary(
                    previous_analysis,
                    {
                        "sentiment": new_sentiment,
                        "readability": new_readability,
                        "content_structure": new_structure,
                        "writing_style": new_style,
                    },
                )
            )

        # Store the progress analysis
        analysis_doc_id = self.document_store.store_analysis_document(
            manuscript_id, f"{stage}_progress_analysis", progress_analysis
        )

        # Store in MongoDB Atlas Vector for easy searching
        doc = Document(
            page_content=json.dumps(progress_analysis, indent=2),
            metadata={
                "type": "progress_analysis",
                "stage": stage,
                "manuscript_id": manuscript_id,
                "analysis_id": analysis_doc_id,
            },
        )

        self.document_store.db.store_documents_with_embeddings("analysis", [doc])

        return progress_analysis

    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze emotional tone and sentiment of the content."""
        # Implementation using LLM
        return {}

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze readability metrics."""
        # Implementation using LLM
        return {}

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze narrative structure."""
        # Implementation using LLM
        return {}

    def _analyze_genre(self, content: str) -> Dict[str, Any]:
        """Determine genre characteristics."""
        # Implementation using LLM
        return {}

    def _identify_themes(self, content: str) -> List[str]:
        """Extract major themes from the content."""
        # Implementation using LLM
        return []

    def _extract_plot_points(self, content: str) -> List[Dict[str, Any]]:
        """Extract major plot points."""
        # Implementation using LLM
        return []

    def _extract_characters(self, content: str) -> List[Dict[str, Any]]:
        """Extract character mentions and descriptions."""
        # Implementation using LLM
        return []

    def _extract_settings(self, content: str) -> List[Dict[str, Any]]:
        """Extract setting descriptions."""
        # Implementation using LLM
        return []

    def _compare_analysis(
        self, previous: Dict[str, Any], current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare previous and current analysis to determine improvements."""
        improvements = {
            "improved_aspects": [],
            "new_aspects": [],
            "changed_aspects": [],
            "improvement_score": 0.0,
        }

        # Skip if either analysis is empty
        if not previous or not current:
            return improvements

        # Track improvement count for scoring
        improvement_count = 0
        total_aspects = 0

        # Compare each key
        all_keys = set(previous.keys()) | set(current.keys())
        for key in all_keys:
            total_aspects += 1

            # New aspect
            if key not in previous and key in current:
                improvements["new_aspects"].append(key)
                improvement_count += 1

            # Changed aspect
            elif key in previous and key in current:
                prev_value = previous[key]
                curr_value = current[key]

                # If values are different
                if prev_value != curr_value:
                    improvements["changed_aspects"].append(
                        {"aspect": key, "previous": prev_value, "current": curr_value}
                    )

                    # Determine if it's an improvement
                    if self._is_improvement(key, prev_value, curr_value):
                        improvements["improved_aspects"].append(key)
                        improvement_count += 1

        # Calculate overall improvement score (0.0-1.0)
        if total_aspects > 0:
            improvements["improvement_score"] = improvement_count / total_aspects

        return improvements

    def _is_improvement(self, aspect: str, previous: Any, current: Any) -> bool:
        """Determine if a change represents an improvement."""
        # For numeric values
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            # For readability scores, higher is generally better
            if "score" in aspect or "count" in aspect:
                return current > previous

        # For text values, longer descriptions might indicate more detail
        if isinstance(previous, str) and isinstance(current, str):
            # If complexity or readability, prefer more moderate values
            if "complexity" in aspect:
                return current.lower() in ["moderate", "balanced"]

            # For most text aspects, more detail is better
            return len(current) > len(previous) * 1.1  # 10% longer

        # For lists, more items might indicate more comprehensive analysis
        if isinstance(previous, list) and isinstance(current, list):
            return len(current) > len(previous)

        # Default - assume change is an improvement
        return True

    def _generate_improvement_summary(
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of improvements from initial to final analysis."""
        # Define prompt for improvement summary
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Manuscript Improvement Analyst. Compare the initial and final analysis of a manuscript
        and summarize the improvements made during the transformation process.
        
        Initial Analysis:
        {initial_analysis}
        
        Final Analysis:
        {final_analysis}
        
        Provide a detailed summary of:
        1. Overall improvement assessment
        2. Most significant improvements
        3. Areas with the most dramatic changes
        4. Remaining opportunities for improvement
        5. Impact on target audience appeal
        
        Format your response as a comprehensive report.
        """
        )

        # Format the analysis data
        initial_str = json.dumps(initial, indent=2)
        final_str = json.dumps(final, indent=2)

        # Create the chain
        chain = (
            {
                "initial_analysis": lambda _: initial_str,
                "final_analysis": lambda _: final_str,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        summary = chain.invoke("Generate improvement summary")

        return {
            "improvement_summary": summary,
            "improvement_metrics": {
                "sentiment_change": self._calculate_metric_change(
                    initial.get("sentiment", {}), final.get("sentiment", {})
                ),
                "readability_change": self._calculate_metric_change(
                    initial.get("readability", {}), final.get("readability", {})
                ),
                "structure_change": self._calculate_metric_change(
                    initial.get("content_structure", {}),
                    final.get("content_structure", {}),
                ),
                "style_change": self._calculate_metric_change(
                    initial.get("writing_style", {}), final.get("writing_style", {})
                ),
            },
        }

    def _calculate_metric_change(
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> float:
        """Calculate the percentage of change between initial and final metrics."""
        if not initial or not final:
            return 0.0

        total_fields = 0
        changed_fields = 0

        for key in initial:
            if key in final and initial[key] != final[key]:
                changed_fields += 1
            total_fields += 1

        return changed_fields / max(total_fields, 1)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()
