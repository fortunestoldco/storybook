from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json
import rechain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplateugh
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from storybook.config import create_llm, get_llm
from storybook.agents.base import BaseAgentumentStore
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore
from storybook.db.mongodb_store import MongoDBStore  # Add missing import
from storybook.tools.document_tools import DocumentTools
class DialogueEnhancer(BaseAgent):
logger = logging.getLogger(__name__)ng dialogue in the manuscript."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
class ContentAnalyzer(BaseAgent):ig)
    """Agent responsible for analyzing manuscript content."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.document_tools = DocumentTools()Any]] = None,
        self.embeddings = OpenAIEmbeddings()nal[Dict[str, Any]] = None,
        self.db = MongoDBStore()        llm_config: Optional[Dict[str, Any]] = None

    def get_tools(self):cript."""
        """Get tools available to this agent."""pt = self.document_store.get_manuscript(manuscript_id)
        return [
            self.document_tools.get_manuscript_tool(), found"}
            self.document_tools.get_manuscript_search_tool(),
        ]        # Extract dialogue sections

    def analyze_content(self, manuscript_id: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze manuscript content with optional runtime LLM configuration."""ot dialogue_sections:
        try:
            # Update LLM if new config providedt_id": manuscript_id,
            if llm_config:ue sections found to enhance.",
                self.llm = create_llm(llm_config)
                
            logger.info(f"Analyzing content for manuscript {manuscript_id}")alyze dialogue quality
            ty(
            # Get manuscript content from document store
            document = self.document_store.get_manuscript(manuscript_id)
            if not document:
                raise ValueError(f"Manuscript {manuscript_id} not found")termine improvement strategies
            improvement_strategies(
            content = document.get("content", "")dialogue_analysis, characters, target_audience, research_insights
            
            # Perform comprehensive content analysis
            analysis = {
                "sentiment": self._analyze_sentiment(content),
                "readability": self._analyze_readability(content),
                "content_structure": self._analyze_structure(content),
                "genre_match": self._analyze_genre(content),
                "themes": self._identify_themes(content),s,
                "key_elements": {
                    "plot_points": self._extract_plot_points(content),
                    "characters": self._extract_characters(content),
                    "settings": self._extract_settings(content),he updates to the manuscript
                }ed_content = self._apply_dialogue_updates(
            }manuscript["content"], updated_sections
            
            return {
                "analysis": analysis,
                "message": "Content analysis completed successfully"document_store.update_manuscript(
            }ontent": updated_content}
        except Exception as e:
            logger.error(f"Error analyzing content for manuscript {manuscript_id}: {e}")
            return {"error": str(e)}        return {

    def analyze_progress(self, manuscript_id: str, previous_analysis: Dict[str, Any], stage: str) -> Dict[str, Any]: on character profiles and target audience.",
        """Analyze progress made after a specific stage of the transformation."""
        manuscript = self.document_store.get_manuscript(manuscript_id)trategies": improvement_strategies,
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")
            return {}    def _extract_dialogue_sections(self, content: str) -> List[Dict[str, Any]]:
alogue."""
        content = manuscript.get("content", "")        # Split the content into paragraphs

        # Define what to analyze based on the stage
        progress_analysis = {}        dialogue_sections = []
t": 0, "end": 0, "content": "", "dialogue_count": 0}
        if stage == "story_arc":
            # Analyze structure improvements
            new_structure = self._analyze_structure(content)
            progress_analysis["structure_improvements"] = self._compare_analysis(
                previous_analysis.get("content_structure", {}), new_structuregue_count = 0
            )
            progress_analysis["content_structure"] = new_structure        for i, para in enumerate(paragraphs):
contains dialogue
        elif stage == "language":
            # Analyze style and readability improvements
            new_style = self._analyze_style(content)
            new_readability = self._analyze_readability(content)                    has_dialogue = True

            progress_analysis["style_improvements"] = self._compare_analysis(
                previous_analysis.get("writing_style", {}), new_style       )  # Approximate dialogue turns
            )
            progress_analysis["readability_improvements"] = self._compare_analysis(
                previous_analysis.get("readability", {}), new_readability If we find dialogue and we're not in a section, start one
            )            if has_dialogue and not in_section:

            progress_analysis["writing_style"] = new_style
            progress_analysis["readability"] = new_readability                dialogue_count = para.count('"') // 2

        elif stage == "complete":ve 3+ paragraphs without dialogue, end it
            # Comprehensive final analysis
            new_sentiment = self._analyze_sentiment(content)
            new_readability = self._analyze_readability(content)
            new_structure = self._analyze_structure(content)ot any(
            new_style = self._analyze_style(content)                        m in paragraphs[i + j] for m in dialogue_markers

            # Compare all metrics with original
            progress_analysis["overall_improvements"] = {
                "sentiment": self._compare_analysis(
                    previous_analysis.get("sentiment", {}), new_sentiment  # End the section
                ),graphs[section_start:i])
                "readability": self._compare_analysis(
                    previous_analysis.get("readability", {}), new_readability  # Only save if there's significant dialogue
                ),
                "structure": self._compare_analysis(
                    previous_analysis.get("content_structure", {}), new_structure          {
                ),_start,
                "style": self._compare_analysis(
                    previous_analysis.get("writing_style", {}), new_style              "content": section_content,
                ),                   "dialogue_count": dialogue_count,
            }                            }

            # Generate improvement summary
            progress_analysis["improvement_summary"] = (
                self._generate_improvement_summary(
                    previous_analysis,
                    {
                        "sentiment": new_sentiment,
                        "readability": new_readability,n_start:])
                        "content_structure": new_structure,
                        "writing_style": new_style,
                    },   "start": section_start,
                )       "end": len(paragraphs) - 1,
            )                    "content": section_content,
 dialogue_count,
        # Store the progress analysis
        analysis_doc_id = self.document_store.store_analysis_document(
            manuscript_id, f"{stage}_progress_analysis", progress_analysis
        )        return dialogue_sections

        # Store in MongoDB Atlas Vector for easy searchingue_quality(
        doc = Document(
            page_content=json.dumps(progress_analysis, indent=2),ons: List[Dict[str, Any]],
            metadata={
                "type": "progress_analysis",al[Dict[str, Any]] = None,
                "stage": stage,
                "manuscript_id": manuscript_id,he manuscript."""
                "analysis_id": analysis_doc_id,le the most dialogue-rich sections (up to 3)
            },ample_sections = sorted(
        )            dialogue_sections, key=lambda x: x["dialogue_count"], reverse=True

        self.document_store.db.store_documents_with_embeddings("analysis", [doc])        sample_text = "\n\n".join([section["content"] for section in sample_sections])

        return progress_analysis        # Prepare character voice information

    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze emotional tone and sentiment of the content."""r:
        # Implementation using LLMharacter_voices.append(f"{character['name']}: {character['voice']}")
        return {}

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze readability metrics."""
        # Implementation using LLM"No character voice information available."
        return {}        )

    def _analyze_structure(self, content: str) -> Dict[str, Any]:e
        """Analyze narrative structure."""
        # Implementation using LLM_audience:
        return {}            audience_context = f"""

    def _analyze_genre(self, content: str) -> Dict[str, Any]:t('demographic', 'General readers')}
        """Determine genre characteristics.""": {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
        # Implementation using LLM
        return {}            Consider whether the dialogue would resonate with this audience.

    def _identify_themes(self, content: str) -> List[str]:
        """Extract major themes from the content."""pt
        # Implementation using LLMChatPromptTemplate.from_template(
        return []            """
ity in the following manuscript sections.
    def _extract_plot_points(self, content: str) -> List[Dict[str, Any]]:
        """Extract major plot points."""n:
        # Implementation using LLMr_voice_info}
        return []        

    def _extract_characters(self, content: str) -> List[Dict[str, Any]]:
        """Extract character mentions and descriptions."""
        # Implementation using LLM_samples}
        return []        

    def _extract_settings(self, content: str) -> List[Dict[str, Any]]:
        """Extract setting descriptions."""ency 
        # Implementation using LLM tags and attribution clarity
        return []        4. Subtext and underlying tension
alogue advances plot or reveals character)
    def _compare_analysis(
        self, previous: Dict[str, Any], current: Dict[str, Any]get audience
    ) -> Dict[str, Any]:
        """Compare previous and current analysis to determine improvements."""onse as a detailed analysis with specific examples.
        improvements = {
            "improved_aspects": [],
            "new_aspects": [],
            "changed_aspects": [],
            "improvement_score": 0.0,hain = (
        }            {
mbda _: character_voice_info,
        # Skip if either analysis is emptybda _: sample_text,
        if not previous or not current:xt": lambda _: audience_context,
            return improvements            }

        # Track improvement count for scoring
        improvement_count = 0rser()
        total_aspects = 0        )

        # Compare each key
        all_keys = set(previous.keys()) | set(current.keys())oke("Analyze dialogue")
        for key in all_keys:
            total_aspects += 1        return {
analysis,
            # New aspect
            if key not in previous and key in current:
                improvements["new_aspects"].append(key)
                improvement_count += 1
nt_strategies(
            # Changed aspect
            elif key in previous and key in current:
                prev_value = previous[key]
                curr_value = current[key]        target_audience: Optional[Dict[str, Any]] = None,
str, Any]] = None,
                # If values are different
                if prev_value != curr_value:
                    improvements["changed_aspects"].append(
                        {"aspect": key, "previous": prev_value, "current": curr_value}PromptTemplate.from_template(
                    )            """
ed on the following dialogue analysis, 
                    # Determine if it's an improvementcript.
                    if self._is_improvement(key, prev_value, curr_value):
                        improvements["improved_aspects"].append(key)
                        improvement_count += 1        {dialogue_analysis}

        # Calculate overall improvement score (0.0-1.0):
        if total_aspects > 0:
            improvements["improvement_score"] = improvement_count / total_aspects        

        return improvements        

    def _is_improvement(self, aspect: str, previous: Any, current: Any) -> bool:
        """Determine if a change represents an improvement."""ategies for:
        # For numeric values
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            # For readability scores, higher is generally better
            if "score" in aspect or "count" in aspect:character motivations
                return current > previous        5. Strengthening dialogue's contribution to plot advancement

        # For text values, longer descriptions might indicate more detail
        if isinstance(previous, str) and isinstance(current, str):
            # If complexity or readability, prefer more moderate valuescific examples of before/after dialogue improvements.
            if "complexity" in aspect:
                return current.lower() in ["moderate", "balanced"]        )

            # For most text aspects, more detail is better
            return len(current) > len(previous) * 1.1  # 10% longer        character_info = []

        # For lists, more items might indicate more comprehensive analysis
        if isinstance(previous, list) and isinstance(current, list):
            return len(current) > len(previous)                char_summary += f"Personality: {character['personality'][:150]}...\n"

        # Default - assume change is an improvementr_summary += f"Motivations: {character['motivations'][:150]}...\n"
        return True            if "voice" in character:
e: {character['voice'][:150]}...\n"
    def _generate_improvement_summary(
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of improvements from initial to final analysis."""
        # Define prompt for improvement summary
        prompt = ChatPromptTemplate.from_template(e_context = ""
            """
        You are a Manuscript Improvement Analyst. Compare the initial and final analysis of a manuscript
        and summarize the improvements made during the transformation process.    Target Audience Information:
        : {target_audience.get('demographic', 'General readers')}
        Initial Analysis:ferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
        {initial_analysis}    - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
        
        Final Analysis: to make dialogue especially appealing to this audience.
        {final_analysis}    """
        
        Provide a detailed summary of:le
        1. Overall improvement assessment
        2. Most significant improvements
        3. Areas with the most dramatic changes
        4. Remaining opportunities for improvement
        5. Impact on target audience appeal    {research_insights.get('market_analysis_summary', 'No market analysis available')}
        
        Format your response as a comprehensive report. Consider these insights when developing the dialogue enhancement strategy.
        """   """
        )

        # Format the analysis data
        initial_str = json.dumps(initial, indent=2)
        final_str = json.dumps(final, indent=2)                "dialogue_analysis": lambda _: dialogue_analysis["analysis"],
_info": lambda _: character_info_text,
        # Create the chainaudience_context": lambda _: audience_context,
        chain = (   "research_context": lambda _: research_context,
            {
                "initial_analysis": lambda _: initial_str,
                "final_analysis": lambda _: final_str, self.llm
            }putParser()
            | prompt
            | self.llm
            | StrOutputParser() Run the chain
        )        strategies = chain.invoke("Generate improvement strategies")

        # Run the chain
        summary = chain.invoke("Generate improvement summary")
_dialogue_sections(
        return {
            "improvement_summary": summary,
            "improvement_metrics": {
                "sentiment_change": self._calculate_metric_change(
                    initial.get("sentiment", {}), final.get("sentiment", {})t_strategies: Dict[str, Any],
                ),
                "readability_change": self._calculate_metric_change(
                    initial.get("readability", {}), final.get("readability", {}) each dialogue section based on the improvement strategies."""
                ),
                "structure_change": self._calculate_metric_change(
                    initial.get("content_structure", {}),
                    final.get("content_structure", {}),y = {
                ),
                "style_change": self._calculate_metric_change(
                    initial.get("writing_style", {}), final.get("writing_style", {})ersonality": character.get("personality", ""),
                ),  "motivations": character.get("motivations", ""),
            },   }
        }            character_summaries[character["name"]] = summary

    def _calculate_metric_change(
        self, initial: Dict[str, Any], final: Dict[str, Any]e_context = ""
    ) -> float:
        """Calculate the percentage of change between initial and final metrics."""
        if not initial or not final:ience: {target_audience.get('demographic', 'General readers')}
            return 0.0            Preferences: {target_audience.get('reading_preferences', {}).get('language', 'No specific preferences')}

        total_fields = 0
        changed_fields = 0        # Process sections (limit to 10 for efficiency)
= []
        for key in initial:sorted(
            if key in final:y=lambda x: x["dialogue_count"], reverse=True
                total_fields += 1
                if initial[key] != final[key]:
                    changed_fields += 1        for section in priority_sections:

        return changed_fields / max(total_fields, 1)            prompt = ChatPromptTemplate.from_template(

    def _get_timestamp(self) -> str:ist. Improve the dialogue in this manuscript section
        """Get current timestamp in ISO format.""" improvement strategies.
        from datetime import datetime            

        return datetime.now().isoformat()            {original_section}

    def _store_analysis_document(self, manuscript_id: str, analysis: Dict[str, Any]) -> str:
        """Store analysis in vector store."""{character_info}
        try:
            doc = Document(
                page_content=json.dumps(analysis, indent=2),trategies}
                metadata={
                    "type": "content_analysis",
                    "manuscript_id": manuscript_id,
                    "timestamp": self._get_timestamp()ce the dialogue in this section while maintaining the story and meaning.
                }ake each character's voice more distinctive and authentic based on their profile.
            )
            return self.document_store.db.store_documents_with_embeddings("analysis", [doc])d dialogue better advances the plot or reveals character.
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")fully enhanced section with all original content, just with improved dialogue.
            return None            Preserve the structure and non-dialogue elements exactly as they are.

            """
            )

            # Extract character names from this section
            character_names = self._extract_character_names(
                section["content"], list(character_summaries.keys())
            )

            # Filter to relevant characters
            relevant_characters = {}
            for name in character_names:
                if name in character_summaries:
                    relevant_characters[name] = character_summaries[name]

            # Format character info
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

            # Create the chain
            chain = (
                {
                    "original_section": lambda _: section["content"],
                    "character_info": lambda _: character_info_text,
                    "improvement_strategies": lambda _: improvement_strategies[
                        "strategies"
                    ],
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            enhanced_section = chain.invoke("Enhance dialogue")

            processed_sections.append(
                {"original": section, "enhanced": enhanced_section}
            )

        return processed_sections

    def _apply_dialogue_updates(
        self, original_content: str, updated_sections: List[Dict[str, Any]]
    ) -> str:
        """Apply dialogue updates to the original content."""
        # Split the content into paragraphs
        paragraphs = original_content.split("\n\n")

        # Track which paragraphs have been updated
        updated_indices = set()

        # Apply updates
        for update in updated_sections:
            original = update["original"]
            start_idx = original["start"]
            end_idx = original["end"]

            # Mark these paragraphs as updated
            for i in range(start_idx, end_idx + 1):
                updated_indices.add(i)

            # Split the enhanced content into paragraphs
            enhanced_paragraphs = update["enhanced"].split("\n\n")

            # If the number of paragraphs has changed dramatically, use a different approach
            if abs(len(enhanced_paragraphs) - (end_idx - start_idx + 1)) > 2:
                # Find unique sentences in the original section
                original_section = "\n\n".join(paragraphs[start_idx : end_idx + 1])
                original_sentences = self._extract_sentences(original_section)

                # Replace the entire section using sentence matching
                new_content = self._replace_section_by_sentences(
                    original_content, original_sentences, update["enhanced"]
                )

                if new_content:
                    return new_content

                # Fallback to paragraph replacement if sentence matching fails

            # Replace the paragraphs
            # Make sure we don't exceed the original content bounds
            replacement_length = min(
                len(enhanced_paragraphs), len(paragraphs) - start_idx
            )
            paragraphs[start_idx : start_idx + replacement_length] = (
                enhanced_paragraphs[:replacement_length]
            )

        # Join the paragraphs back into content
        return "\n\n".join(paragraphs)

    def _extract_character_names(self, text: str, known_characters: List[str]) -> List[str]:
        """Extract character names mentioned in the text."""
        # Check for exact matches of known characters
        mentioned_characters = [char for char in known_characters if char.lower() in text.lower()]
        return mentioned_characters

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter out empty strings
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

        # Find the start of the first sentence
        first_sentence = re.escape(original_sentences[0])
        start_match = re.search(first_sentence, original_content)
        if not start_match:
            return None

        # Find the end of the last sentence
        last_sentence = re.escape(original_sentences[-1])
        end_match = re.search(last_sentence, original_content)
        if not end_match:
            return None

        # Calculate the replacement indices
        start_idx = start_match.start()
        end_idx = end_match.end()

        # Replace the content
        return (
            original_content[:start_idx] + enhanced_content + original_content[end_idx:]
        )
