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

# Local imports
from storybook.agents.base import BaseAgent
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore
from storybook.tools.document_tools import DocumentTools

logger = logging.getLogger(__name__)

class DialogueEnhancer(BaseAgent):
    """Agent responsible for enhancing dialogue in manuscripts."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.document_tools = DocumentTools()

    def get_tools(self) -> List[Any]:
        """Get tools available to this agent."""
        return [
            self.document_tools.get_manuscript_tool(),
            self.document_tools.get_manuscript_search_tool()
        ]

    def enhance_dialogue(
        self,
        manuscript_id: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance dialogue in the manuscript."""
        try:
            if llm_config:
                self.llm = create_llm(llm_config)

            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # ... rest of implementation ...
            return {
                "manuscript_id": manuscript_id,
                "message": "Dialogue enhancement complete",
                "enhanced_content": "enhanced content here"
            }

        except Exception as e:
            logger.error(f"Error in enhance_dialogue: {str(e)}")
            return self.handle_error(e)

class ContentAnalyzer(BaseAgent):
    """Agent responsible for analyzing manuscript content."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
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
            if llm_config:
                self.llm = create_llm(llm_config)
                
            logger.info(f"Analyzing content for manuscript {manuscript_id}")
            document = self.document_store.get_manuscript(manuscript_id)
            if not document:
                raise ValueError(f"Manuscript {manuscript_id} not found")
            
            content = document.get("content", "")
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
        progress_analysis = {}
        if stage == "story_arc":
            new_structure = self._analyze_structure(content)
            progress_analysis["structure_improvements"] = self._compare_analysis(
                previous_analysis.get("content_structure", {}), new_structure
            )
            progress_analysis["content_structure"] = new_structure
        elif stage == "language":
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
            new_sentiment = self._analyze_sentiment(content)
            new_readability = self._analyze_readability(content)
            new_structure = self._analyze_structure(content)
            new_style = self._analyze_style(content)
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

        analysis_doc_id = self.document_store.store_analysis_document(
            manuscript_id, f"{stage}_progress_analysis", progress_analysis
        )

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
        return {}

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze readability metrics."""
        return {}

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze narrative structure."""
        return {}

    def _analyze_genre(self, content: str) -> Dict[str, Any]:
        """Determine genre characteristics."""
        return {}

    def _identify_themes(self, content: str) -> List[str]:
        """Extract major themes from the content."""
        return []

    def _extract_plot_points(self, content: str) -> List[Dict[str, Any]]:
        """Extract major plot points."""
        return []

    def _extract_characters(self, content: str) -> List[Dict[str, Any]]:
        """Extract character mentions and descriptions."""
        return []

    def _extract_settings(self, content: str) -> List[Dict[str, Any]]:
        """Extract setting descriptions."""
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

        if not previous or not current:
            return improvements

        improvement_count = 0
        total_aspects = 0

        all_keys = set(previous.keys()) | set(current.keys())
        for key in all_keys:
            total_aspects += 1
            if key not in previous and key in current:
                improvements["new_aspects"].append(key)
                improvement_count += 1
            elif key in previous and key in current:
                prev_value = previous[key]
                curr_value = current[key]
                if prev_value != curr_value:
                    improvements["changed_aspects"].append(
                        {"aspect": key, "previous": prev_value, "current": curr_value}
                    )
                    if self._is_improvement(key, prev_value, curr_value):
                        improvements["improved_aspects"].append(key)
                        improvement_count += 1

        if total_aspects > 0:
            improvements["improvement_score"] = improvement_count / total_aspects

        return improvements

    def _is_improvement(self, aspect: str, previous: Any, current: Any) -> bool:
        """Determine if a change represents an improvement."""
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            if "score" in aspect or "count" in aspect:
                return current > previous

        if isinstance(previous, str) and isinstance(current, str):
            if "complexity" in aspect:
                return current.lower() in ["moderate", "balanced"]
            return len(current) > len(previous) * 1.1

        if isinstance(previous, list) and isinstance(current, list):
            return len(current) > len(previous)

        return True

    def _generate_improvement_summary(
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of improvements from initial to final analysis."""
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
        """
        )

        initial_str = json.dumps(initial, indent=2)
        final_str = json.dumps(final, indent=2)

        chain = (
            {
                "initial_analysis": lambda _: initial_str,
                "final_analysis": lambda _: final_str,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

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
            if key in final:
                total_fields += 1
                if initial[key] != final[key]:
                    changed_fields += 1

        return changed_fields / max(total_fields, 1)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _store_analysis_document(self, manuscript_id: str, analysis: Dict[str, Any]) -> str:
        """Store analysis in vector store."""
        try:
            doc = Document(
                page_content=json.dumps(analysis, indent=2),
                metadata={
                    "type": "content_analysis",
                    "manuscript_id": manuscript_id,
                    "timestamp": self._get_timestamp()
                }
            )
            return self.document_store.db.store_documents_with_embeddings("analysis", [doc])
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            return None

    def _extract_dialogue_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract dialogue sections from the manuscript."""
        paragraphs = content.split("\n\n")
        dialogue_sections = []
        in_section = False
        section_start = 0
        dialogue_count = 0

        for i, para in enumerate(paragraphs):
            has_dialogue = any(marker in para for marker in ['"', "'"])
            if has_dialogue:
                dialogue_count += para.count('"') // 2
                if not in_section:
                    in_section = True
                    section_start = i
            elif in_section:
                if i - section_start > 2:
                    section_content = "\n\n".join(paragraphs[section_start:i])
                    dialogue_sections.append({
                        "start": section_start,
                        "end": i - 1,
                        "content": section_content,
                        "dialogue_count": dialogue_count,
                    })
                    in_section = False
                    dialogue_count = 0

        if in_section:
            section_content = "\n\n".join(paragraphs[section_start:])
            dialogue_sections.append({
                "start": section_start,
                "end": len(paragraphs) - 1,
                "content": section_content,
                "dialogue_count": dialogue_count,
            })

        return dialogue_sections

    def _enhance_dialogue_sections(
        self,
        dialogue_sections: List[Dict[str, Any]],
        improvement_strategies: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Enhance the dialogue in each dialogue section based on the improvement strategies."""
        processed_sections = []

        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            """

        research_context = ""
        if research_insights:
            research_context = f"""
            Research Insights:
            - Market Analysis Summary: {research_insights.get('market_analysis_summary', 'No market analysis available')}
            """

        priority_sections = sorted(
            dialogue_sections, key=lambda x: x["dialogue_count"], reverse=True
        )[:10]

        for section in priority_sections:
            prompt = ChatPromptTemplate.from_template(
                """
                You are a Dialogue Enhancement Specialist. Improve the dialogue in this manuscript section
                based on the following improvement strategies.
                
                Original Section:
                {original_section}
                
                Character Information:
                {character_info}
                
                Improvement Strategies:
                {improvement_strategies}
                
                Audience Context:
                {audience_context}
                
                Research Context:
                {research_context}
                
                Enhance the dialogue in this section while maintaining the story and meaning.
                Make each character's voice more distinctive and authentic based on their profile.
                Ensure the enhanced dialogue better advances the plot or reveals character.
                Return the fully enhanced section with all original content, just with improved dialogue.
                Preserve the structure and non-dialogue elements exactly as they are.
                """
            )

            character_names = self._extract_character_names(
                section["content"], list(character_summaries.keys())
            )

            relevant_characters = {}
            for name in character_names:
                if name in character_summaries:
                    relevant_characters[name] = character_summaries[name]

            char_info = []
            for name, info in relevant_characters.items():
                char_info.append(f"Character: {name}")
                char_info.append(f"Voice: {info['voice'][:200]}")
                char_info.append(f"Personality: {info['personality'][:200]}")
                char_info.append(f"Motivations: {info['motivations'][:200]}")
                char_info.append("")

            character_info_text = (
                "\n".join(char_info)
                if char_info
                else "No specific character information available."
            )

            chain = (
                {
                    "original_section": lambda _: section["content"],
                    "character_info": lambda _: character_info_text,
                    "improvement_strategies": lambda _: improvement_strategies[
                        "strategies"
                    ],
                    "audience_context": lambda _: audience_context,
                    "research_context": lambda _: research_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            enhanced_section = chain.invoke("Enhance dialogue")

            processed_sections.append(
                {"original": section, "enhanced": enhanced_section}
            )

        return processed_sections

    def _apply_dialogue_updates(
        self, original_content: str, updated_sections: List[Dict[str, Any]]
    ) -> str:
        """Apply dialogue updates to the original content."""
        paragraphs = original_content.split("\n\n")
        updated_indices = set()

        for update in updated_sections:
            original = update["original"]
            start_idx = original["start"]
            end_idx = original["end"]

            for i in range(start_idx, end_idx + 1):
                updated_indices.add(i)

            enhanced_paragraphs = update["enhanced"].split("\n\n")

            if abs(len(enhanced_paragraphs) - (end_idx - start_idx + 1)) > 2:
                original_section = "\n\n".join(paragraphs[start_idx : end_idx + 1])
                original_sentences = self._extract_sentences(original_section)
                new_content = self._replace_section_by_sentences(
                    original_content, original_sentences, update["enhanced"]
                )

                if new_content:
                    return new_content

            replacement_length = min(
                len(enhanced_paragraphs), len(paragraphs) - start_idx
            )
            paragraphs[start_idx : start_idx + replacement_length] = (
                enhanced_paragraphs[:replacement_length]
            )

        return "\n\n".join(paragraphs)

    def _extract_character_names(self, text: str, known_characters: List[str]) -> List[str]:
        """Extract character names mentioned in the text."""
        mentioned_characters = [char for char in known_characters if char.lower() in text.lower()]
        return mentioned_characters

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s]

    def _replace_section_by_sentences(
        self,
        original_content: str,
        original_sentences: List[str],
        enhanced_content: str,
    ) -> Optional[str]:
        """Replace a section by finding sentence boundaries in the original content."""
        if not original_sentences:
            return None

        first_sentence = re.escape(original_sentences[0])
        start_match = re.search(first_sentence, original_content)
        if not start_match:
            return None

        last_sentence = re.escape(original_sentences[-1])
        end_match = re.search(last_sentence, original_content)
        if not end_match:
            return None

        start_idx = start_match.start()
        end_idx = end_match.end()

        return (
            original_content[:start_idx] + enhanced_content + original_content[end_idx:]
        )
