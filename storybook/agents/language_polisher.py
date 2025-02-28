from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional
import logging
import json
import re

# Third-party imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Local imports
from storybook.agents.base import BaseAgent
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class LanguagePolisher(BaseAgent):
    """Agent responsible for polishing language and style."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()

    def polish_language(
        self,
        manuscript_id: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Polish language and style in the manuscript."""
        try:
            # Update LLM if new config provided
            if llm_config:
                self.llm = create_llm(llm_config)

            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # Add audience context if available
            audience_context = ""
            if target_audience:
                audience_context = f"""
                Target Audience: {target_audience.get('demographic', 'General readers')}
                Reading Level: {target_audience.get('reading_level', 'Standard')}
                Style Preferences: {target_audience.get('style_preferences', 'Not specified')}
                """

            # Create prompt for language polishing
            prompt = ChatPromptTemplate.from_template(
                """
                You are an expert Language Polisher. Review and enhance the following manuscript text,
                focusing on clarity, style, and engagement.

                {audience_context}

                Original Text:
                {manuscript_text}

                Please polish the language focusing on:
                1. Clarity and readability
                2. Sentence structure and flow
                3. Word choice and vocabulary
                4. Style consistency
                5. Grammar and punctuation
                6. Voice and tone

                Provide:
                1. The polished text
                2. A summary of changes made
                3. Style recommendations
                """
            )

            # Create the chain
            chain = (
                {
                    "manuscript_text": lambda _: manuscript["content"],
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            result = chain.invoke("Polish manuscript language")

            # Update manuscript with polished content
            self.document_store.update_manuscript(
                manuscript_id,
                {"content": result}
            )

            return {
                "manuscript_id": manuscript_id,
                "message": "Language polishing complete",
                "polished_content": result
            }

        except Exception as e:
            logger.error(f"Error in polish_language: {e}")
            return self.handle_error(e)

    def _analyze_language_style(
        self, content: str, target_audience: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze the language and writing style of the manuscript.""" 
        # Sample representative sections
        sample_size = min(5000, len(content) // 4)
        beginning = content[:sample_size]
        middle_start = max(0, (len(content) // 2) - (sample_size // 2))
        middle = content[middle_start : middle_start + sample_size]
        end_start = max(0, len(content) - sample_size)
        end = content[end_start:]

        # Combine samples
        sample = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            
            Consider whether the language style is appropriate for this audience.
            """

        # Create prompt for style analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Literary Style Analyst specializing in prose evaluation. Analyze the writing style and language
        of this manuscript based on the provided excerpts.
        
        Manuscript Excerpts:
        {manuscript_excerpts}
        
        {audience_context}
        
        Analyze the following aspects of writing style:
        
        1. Voice and Point of View: First person, third person, etc. and consistency
        2. Tone: Formal/informal, serious/humorous, emotional/detached
        3. Sentence Structure: Variety, complexity, flow
        4. Vocabulary Level: Simple/complex, specific/general, contemporary/archaic
        5. Literary Devices: Use of metaphor, simile, imagery, etc.
        6. Dialogue Style: How characters speak, dialogue tags, etc.
        7. Description Quality: Sensory details, immersion, showing vs. telling
        8. Prose Rhythm: Pacing of sentences, paragraph structure
        9. Strengths: What aspects of the writing style are most effective
        10. Weaknesses: What aspects could be improved
        11. Audience Appropriateness: How well the style suits the target audience
        
        Provide a detailed analysis with specific examples from the text.
        """
        )

        # Create the chain
        chain = (
            {
                "manuscript_excerpts": lambda _: sample,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        style_analysis = chain.invoke("Analyze writing style")

        # Parse the analysis into sections
        import re

        # Extract key sections
        voice_match = re.search(
            r"Voice and Point of View:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,
        )
        tone_match = re.search(
            r"Tone:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        sentence_match = re.search(
            r"Sentence Structure:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,
        )
        vocab_match = re.search(
            r"Vocabulary Level:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        lit_devices_match = re.search(
            r"Literary Devices:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        dialogue_match = re.search(
            r"Dialogue Style:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        description_match = re.search(
            r"Description Quality:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,
        )
        rhythm_match = re.search(
            r"Prose Rhythm:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        strengths_match = re.search(
            r"Strengths:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        weaknesses_match = re.search(
            r"Weaknesses:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)", style_analysis, re.DOTALL
        )
        audience_match = re.search(
            r"Audience Appropriateness:?\s*(.*?)(?=\n\n|\n\d+\.|\Z)",
            style_analysis,
            re.DOTALL,
        )

        # Compile the structured analysis
        return {
            "full_analysis": style_analysis,
            "voice_and_pov": voice_match.group(1).strip() if voice_match else "",
            "tone": tone_match.group(1).strip() if tone_match else "",
            "sentence_structure": (
                sentence_match.group(1).strip() if sentence_match else ""
            ),
            "vocabulary_level": vocab_match.group(1).strip() if vocab_match else "",
            "literary_devices": (
                lit_devices_match.group(1).strip() if lit_devices_match else ""
            ),
            "dialogue_style": dialogue_match.group(1).strip() if dialogue_match else "",
            "description_quality": (
                description_match.group(1).strip() if description_match else ""
            ),
            "prose_rhythm": rhythm_match.group(1).strip() if rhythm_match else "",
            "strengths": strengths_match.group(1).strip() if strengths_match else "",
            "weaknesses": weaknesses_match.group(1).strip() if weaknesses_match else "",
            "audience_appropriateness": (
                audience_match.group(1).strip() if audience_match else ""
            ),
        }

    def _identify_improvement_areas(
        self,
        content: str,
        style_analysis: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Identify specific areas for language improvement.""" 
        # Extract weaknesses from the style analysis
        weaknesses = style_analysis.get("weaknesses", "")

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            - Reading Level: {style_analysis.get('audience_appropriateness', 'Unknown')}
            
            Focus on improvements that would make the language more appealing to this audience.
            """

        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider how language improvements could align with successful books in this market.
            """

        # Create prompt for identifying improvement areas
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Literary Style Enhancement Specialist. Based on the style analysis, identify specific 
        aspects of the manuscript's language that could be improved.
        
        Style Analysis Strengths:
        {style_strengths}
        
        Style Analysis Weaknesses:
        {style_weaknesses}
        
        {audience_context}
        
        {research_context}
        
        Identify 5-7 specific improvement areas, such as:
        
        1. Sentence variety (e.g., too many sentences with similar structure)
        2. Overused words or phrases
        3. Weak descriptions lacking sensory details
        4. Telling instead of showing
        5. Passive voice overuse
        6. Dialogue that sounds unnatural
        7. Inconsistent tone
        8. Too much/little exposition
        
        For each improvement area:
        - Clearly describe the issue
        - Explain its impact on reader engagement
        - Provide a specific example from the style analysis
        - Suggest an approach for addressing it
        
        Focus on improvements that would have the greatest impact on reader experience,
        especially for the target audience.
        """
        )

        # Create the chain
        chain = (
            {
                "style_strengths": lambda _: style_analysis.get(
                    "strengths", "Not specified"
                ),
                "style_weaknesses": lambda _: weaknesses,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        improvement_areas = chain.invoke("Identify improvement areas")

        # Parse the areas into sections
        import re

        # Try to extract individual improvement areas
        area_blocks = re.split(r"(?:\n|^)(\d+\.\s+)", improvement_areas)

        # Process the blocks
        structured_areas = []
        current_area = {}

        for i, block in enumerate(area_blocks):
            # If this is a number marker, start a new area
            if re.match(r"\d+\.\s+", block):
                if current_area and "name" in current_area:
                    structured_areas.append(current_area)
                current_area = {"number": block.strip()}
            # If we have a current area and this isn't a number marker, add the content
            elif current_area:
                # First block after number is the title/name
                if "name" not in current_area:
                    current_area["name"] = block.strip().split("\n")[0].strip() if block.strip() else ""
                    # Extract the rest of the content
                    content_lines = block.strip().split("\n")[1:] if len(block.strip().split("\n")) > 1 else []  # Fixed parentheses
                    current_area["description"] = "\n".join(content_lines).strip()
                else:
                    # Append to existing description
                    if "description" in current_area:
                        current_area["description"] += "\n" + block.strip()
                    else:
                        current_area["description"] = block.strip()

        # Add the last area if it exists
        if current_area and "name" in current_area:
            structured_areas.append(current_area)

        return {"full_analysis": improvement_areas, "areas": structured_areas}

    def _polish_sections(
        self,
        content: str,
        improvement_areas: Dict[str, Any],
        style_analysis: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Polish selected sections of the manuscript based on identified improvement areas.""" 
        # Break the content into paragraphs
        paragraphs = content.split("\n\n")

        # We'll enhance a selection of paragraphs rather than the entire manuscript
        # Select paragraphs for each improvement area
        improvement_types = [
            area.get("name", "") for area in improvement_areas.get("areas", [])
        ]

        # Map improvement types to paragraph selection criteria
        selection_criteria = {
            "sentence variety": lambda p: len(p.split(".")) > 4 and len(p) > 200,
            "overused words": lambda p: len(p) > 200,
            "weak descriptions": lambda p: "saw" in p.lower()
            or "looked" in p.lower()
            or "seemed" in p.lower(),
            "telling": lambda p: "felt" in p.lower()
            or "thought" in p.lower()
            or "realized" in p.lower(),
            "passive voice": lambda p: " was " in p.lower() and " by " in p.lower(),
            "dialogue": lambda p: '"' in p or "'" in p,
            "tone": lambda p: len(p) > 200,
            "exposition": lambda p: len(p) > 300,
        }

        # Select paragraphs for improvement
        paragraphs_to_improve = []

        for i, para in enumerate(paragraphs):
            # Skip very short paragraphs
            if len(para) < 50:
                continue

            # Check against our criteria
            for imp_type, criterion in selection_criteria.items():
                if any(
                    imp_type.lower() in area.lower() for area in improvement_types
                ) and criterion(para):
                    paragraphs_to_improve.append((i, para, imp_type))
                    break

        # Limit the number of paragraphs to improve
        max_improvements = min(10, len(paragraphs_to_improve))
        # Prioritize by spreading throughout the manuscript
        step = (
            len(paragraphs_to_improve) // max_improvements
            if len(paragraphs_to_improve) > max_improvements
            else 1
        )
        selected_paragraphs = [
            paragraphs_to_improve[i] for i in range(0, len(paragraphs_to_improve), step)
        ][:max_improvements]

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'No specific preferences')}
            
            Ensure the enhancements will appeal to this audience.
            """

        # Improve each selected paragraph
        improved_sections = []

        for idx, para, imp_type in selected_paragraphs:
            # Create prompt for polishing this paragraph
            prompt = ChatPromptTemplate.from_template(
                """
            You are a Literary Style Enhancer. Improve the following paragraph from a manuscript,
            focusing particularly on {improvement_type}.
            
            Original Paragraph:
            {original_paragraph}
            
            Writing Style Notes:
            Voice: {voice}
            Tone: {tone}
            Vocabulary Level: {vocabulary}
            
            {audience_context}
            
            Enhance this paragraph to address the {improvement_type} issue while:
            1. Maintaining the same story information and meaning
            2. Preserving the author's voice and overall tone
            3. Keeping approximately the same length
            4. Making the language more engaging and vivid
            
            Return ONLY the improved paragraph, with no additional comments.
            """
            )

            # Create the chain
            chain = (
                {
                    "improvement_type": lambda _: imp_type,
                    "original_paragraph": lambda _: para,
                    "voice": lambda _: style_analysis.get(
                        "voice_and_pov", "Not specified"
                    ),
                    "tone": lambda _: style_analysis.get("tone", "Not specified"),
                    "vocabulary": lambda _: style_analysis.get(
                        "vocabulary_level", "Not specified"
                    ),
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            improved_para = chain.invoke(f"Improve paragraph with {imp_type} issue")

            # Store the improvement
            improved_sections.append(
                {
                    "original": para,
                    "improved": improved_para,
                    "improvement_type": imp_type,
                }
            )

            # Update the paragraph in the manuscript
            paragraphs[idx] = improved_para

        # Recombine the paragraphs
        updated_content = "\n\n".join(paragraphs)

        return updated_content, improved_sections

    def _extract_style_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract common style patterns from the text.""" 
        # Implementation for style pattern extraction
        pass

    def _apply_style_rules(
        self,
        content: str,
        style_rules: Dict[str, Any]
    ) -> str:
        """Apply predefined style rules to the text.""" 
        # Implementation for applying style rules
        pass

    def _analyze_block(self, block: str) -> Dict[str, Any]:
        """Analyze a block of text for style and language patterns."""
        try:
            if len(block.strip().split("\n")) > 1:  # Fixed syntax error here
                # Process multi-line block
                return {
                    "length": len(block),
                    "sentences": len(re.findall(r'[.!?]+', block)),
                    "paragraphs": len(block.strip().split("\n\n")),
                    "patterns": self._extract_patterns(block)
                }
            return {}
        except Exception as e:
            logger.error(f"Error analyzing block: {str(e)}")
            return {}

    def _extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract common language patterns from text."""
        patterns = {
            "repetitive_words": [],
            "complex_phrases": [],
            "weak_constructions": []
        }
        try:
            # Extract patterns using regex
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Find repetitive words
            patterns["repetitive_words"] = [
                word for word, count in word_freq.items()
                if count > 3 and len(word) > 3
            ]

            return patterns
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return patterns
