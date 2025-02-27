from typing import Dict, List, Any, Optional
import logging
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate

from storybook.config import get_llm
from storybook.db.document_store import DocumentStore
from storybook.tools.document_tools import DocumentTools

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Agent responsible for analyzing manuscript content using NLP techniques."""

    def __init__(self):
        self.llm = get_llm(
            temperature=0.4, use_replicate=True
        )  # Lower temperature for more consistent analysis
        self.document_store = DocumentStore()
        self.document_tools = DocumentTools()

    def get_tools(self):
        """Get tools available to this agent."""
        return [
            self.document_tools.get_manuscript_tool(),
            self.document_tools.get_manuscript_search_tool(),
        ]

    def analyze_content(self, manuscript_id: str) -> Dict[str, Any]:
        """Perform NLP analysis on the manuscript content."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")
            return {"status": "error", "message": "Manuscript not found"}

        content = manuscript.get("content", "")
        title = manuscript.get("title", "Untitled")

        # Break down the analysis into several components
        sentiment_analysis = self._analyze_sentiment(content)
        readability_analysis = self._analyze_readability(content)
        structure_analysis = self._analyze_structure(content)
        style_analysis = self._analyze_style(content)
        genre_analysis = self._analyze_genre_match(content, title)

        # Compile the complete analysis
        complete_analysis = {
            "manuscript_id": manuscript_id,
            "title": title,
            "sentiment": sentiment_analysis,
            "readability": readability_analysis,
            "content_structure": structure_analysis,
            "writing_style": style_analysis,
            "genre_match": genre_analysis,
            "timestamp": self._get_timestamp(),
        }

        # Store the analysis in the database
        self.document_store.store_analysis_document(
            manuscript_id, "initial_content_analysis", complete_analysis
        )

        return {
            "manuscript_id": manuscript_id,
            "status": "success",
            "message": "Completed NLP analysis of manuscript content.",
            "analysis": complete_analysis,
        }

    def analyze_progress(
        self, manuscript_id: str, previous_analysis: Dict[str, Any], stage: str
    ) -> Dict[str, Any]:
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
        self.document_store.store_analysis_document(
            manuscript_id, f"{stage}_progress_analysis", progress_analysis
        )

        return progress_analysis

    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze the sentiment and emotional content of the manuscript."""
        # Use Replicate LLM for sentiment analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Text Analysis Specialist. Analyze the sentiment and emotional content of this manuscript excerpt.
        
        Text: {text_sample}
        
        Provide a detailed analysis of:
        1. Overall emotional tone (positive, negative, neutral, mixed)
        2. Dominant emotions expressed
        3. Emotional variations throughout the text
        4. Intensity of emotional content (1-10 scale)
        
        Format your response as a JSON object with keys: "overall_tone", "dominant_emotions", 
        "emotional_variations", and "intensity_score".
        """
        )

        # Create the chain
        chain = (
            {"text_sample": lambda _: content[:5000]}  # Sample first 5000 chars
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis_str = chain.invoke("Analyze sentiment")

        # Parse the JSON response
        try:
            return json.loads(analysis_str)
        except json.JSONDecodeError:
            # Extract values using regex if JSON parsing fails
            result = {}

            tone_match = re.search(
                r'overall_tone"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if tone_match:
                result["overall_tone"] = tone_match.group(1).strip()

            emotions_match = re.search(
                r'dominant_emotions"?\s*:?\s*(?:\[([^\]]+)\]|"([^"]+)")', analysis_str
            )
            if emotions_match:
                emotions_str = emotions_match.group(1) or emotions_match.group(2)
                result["dominant_emotions"] = [
                    e.strip().strip("\"'") for e in emotions_str.split(",")
                ]

            intensity_match = re.search(r'intensity_score"?\s*:?\s*(\d+)', analysis_str)
            if intensity_match:
                result["intensity_score"] = int(intensity_match.group(1))

            return result

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze the readability and complexity of the manuscript."""
        # Calculate basic metrics manually
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        paragraphs = content.split("\n\n")

        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_paragraph_length = len(sentences) / max(len(paragraphs), 1)

        # Define prompt for readability analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Readability Analysis Specialist. Analyze the readability of this manuscript excerpt.
        
        Text sample: {text_sample}
        
        Basic metrics:
        - Word count: {word_count}
        - Average word length: {avg_word_length:.2f} characters
        - Average sentence length: {avg_sentence_length:.2f} words
        - Average paragraph length: {avg_paragraph_length:.2f} sentences
        
        Provide a detailed analysis of:
        1. Approximate reading grade level (e.g., 6th grade, college level)
        2. Vocabulary complexity (simple, moderate, advanced)
        3. Sentence structure complexity
        4. Overall readability assessment
        
        Format your response as a JSON object with keys: "grade_level", "vocabulary_complexity", 
        "sentence_complexity", and "overall_readability".
        """
        )

        # Create the chain
        chain = (
            {
                "text_sample": lambda _: content[:3000],
                "word_count": lambda _: len(words),
                "avg_word_length": lambda _: avg_word_length,
                "avg_sentence_length": lambda _: avg_sentence_length,
                "avg_paragraph_length": lambda _: avg_paragraph_length,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis_str = chain.invoke("Analyze readability")

        # Parse the JSON response
        try:
            llm_analysis = json.loads(analysis_str)

            # Combine calculated metrics with LLM analysis
            return {
                "word_count": len(words),
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length,
                "avg_paragraph_length": avg_paragraph_length,
                **llm_analysis,
            }
        except json.JSONDecodeError:
            # Extract values using regex if JSON parsing fails
            result = {
                "word_count": len(words),
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length,
                "avg_paragraph_length": avg_paragraph_length,
            }

            grade_match = re.search(
                r'grade_level"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if grade_match:
                result["grade_level"] = grade_match.group(1).strip()

            vocab_match = re.search(
                r'vocabulary_complexity"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if vocab_match:
                result["vocabulary_complexity"] = vocab_match.group(1).strip()

            readability_match = re.search(
                r'overall_readability"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if readability_match:
                result["overall_readability"] = readability_match.group(1).strip()

            return result

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the narrative structure of the manuscript."""
        # Define prompt for structure analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Narrative Structure Analyst. Analyze the structure of this manuscript excerpt.
        
        Text sample: {text_sample}
        
        Provide a detailed analysis of:
        1. Narrative structure (linear, non-linear, episodic, etc.)
        2. Pacing (identify fast-paced and slow-paced sections)
        3. Plot elements (setup, inciting incident, rising action, etc.)
        4. Chapter organization and transitions
        5. Balance between dialogue, action, and description
        
        Format your response as a JSON object with keys: "narrative_structure", "pacing_analysis", 
        "plot_elements", "chapter_organization", and "content_balance".
        """
        )

        # Create the chain
        chain = (
            {"text_sample": lambda _: content[:10000]}  # Sample first 10000 chars
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis_str = chain.invoke("Analyze structure")

        # Parse the JSON response
        try:
            return json.loads(analysis_str)
        except json.JSONDecodeError:
            # Extract values using regex if JSON parsing fails
            result = {}

            structure_match = re.search(
                r'narrative_structure"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if structure_match:
                result["narrative_structure"] = structure_match.group(1).strip()

            pacing_match = re.search(
                r'pacing_analysis"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if pacing_match:
                result["pacing_analysis"] = pacing_match.group(1).strip()

            balance_match = re.search(
                r'content_balance"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if balance_match:
                result["content_balance"] = balance_match.group(1).strip()

            return result

    def _analyze_style(self, content: str) -> Dict[str, Any]:
        """Analyze the writing style of the manuscript."""
        # Define prompt for style analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Literary Style Analyst. Analyze the writing style of this manuscript excerpt.
        
        Text sample: {text_sample}
        
        Provide a detailed analysis of:
        1. Voice (first person, third person, etc.)
        2. Tone (formal, informal, serious, humorous, etc.)
        3. Distinctive stylistic elements (sentence structure, word choice, literary devices)
        4. Dialogue style and effectiveness
        5. Descriptive techniques
        
        Format your response as a JSON object with keys: "voice", "tone", "distinctive_elements", 
        "dialogue_style", and "descriptive_techniques".
        """
        )

        # Create the chain
        chain = (
            {"text_sample": lambda _: content[:7000]}  # Sample first 7000 chars
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis_str = chain.invoke("Analyze style")

        # Parse the JSON response
        try:
            return json.loads(analysis_str)
        except json.JSONDecodeError:
            # Extract values using regex if JSON parsing fails
            result = {}

            voice_match = re.search(r'voice"?\s*:?\s*"?([^",\}]+)"?', analysis_str)
            if voice_match:
                result["voice"] = voice_match.group(1).strip()

            tone_match = re.search(r'tone"?\s*:?\s*"?([^",\}]+)"?', analysis_str)
            if tone_match:
                result["tone"] = tone_match.group(1).strip()

            dialogue_match = re.search(
                r'dialogue_style"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if dialogue_match:
                result["dialogue_style"] = dialogue_match.group(1).strip()

            return result

    def _analyze_genre_match(self, content: str, title: str) -> Dict[str, Any]:
        """Analyze how well the manuscript matches common genre conventions."""
        # Define prompt for genre match analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Genre Analysis Specialist. Based on this manuscript excerpt and title,
        determine the likely genre(s) and how well it adheres to genre conventions.
        
        Title: {title}
        Text sample: {text_sample}
        
        Provide a detailed analysis of:
        1. Primary genre and potential subgenres
        2. Key genre elements present
        3. Genre expectations met or subverted
        4. Cross-genre elements, if any
        5. Target audience based on genre elements
        
        Format your response as a JSON object with keys: "primary_genre", "subgenres", "genre_elements", 
        "genre_expectations", "cross_genre_elements", and "genre_target_audience".
        """
        )

        # Create the chain
        chain = (
            {
                "title": lambda _: title,
                "text_sample": lambda _: content[:8000],  # Sample first 8000 chars
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis_str = chain.invoke("Analyze genre match")

        # Parse the JSON response
        try:
            return json.loads(analysis_str)
        except json.JSONDecodeError:
            # Extract values using regex if JSON parsing fails
            result = {}

            genre_match = re.search(
                r'primary_genre"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if genre_match:
                result["primary_genre"] = genre_match.group(1).strip()

            subgenres_match = re.search(
                r'subgenres"?\s*:?\s*(?:\[([^\]]+)\]|"([^"]+)")', analysis_str
            )
            if subgenres_match:
                subgenres_str = subgenres_match.group(1) or subgenres_match.group(2)
                result["subgenres"] = [
                    g.strip().strip("\"'") for g in subgenres_str.split(",")
                ]

            audience_match = re.search(
                r'genre_target_audience"?\s*:?\s*"?([^",\}]+)"?', analysis_str
            )
            if audience_match:
                result["genre_target_audience"] = audience_match.group(1).strip()

            return result

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
