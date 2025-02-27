from typing import Dict, List, Any, Optional
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from storybook.config import get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)

class DialogueEnhancer:
    """Agent responsible for enhancing dialogue in the manuscript."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)
        self.document_store = DocumentStore()
    
    def enhance_dialogue(self, manuscript_id: str, characters: List[Dict[str, Any]], 
                        target_audience: Optional[Dict[str, Any]] = None,
                        research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance dialogue throughout the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}
        
        # Extract dialogue sections
        dialogue_sections = self._extract_dialogue_sections(manuscript["content"])
        
        if not dialogue_sections:
            return {
                "manuscript_id": manuscript_id,
                "message": "No significant dialogue sections found to enhance."
            }
        
        # Analyze dialogue quality
        dialogue_analysis = self._analyze_dialogue_quality(
            dialogue_sections, 
            characters, 
            target_audience
        )
        
        # Determine improvement strategies
        improvement_strategies = self._determine_improvement_strategies(
            dialogue_analysis, 
            characters,
            target_audience,
            research_insights
        )
        
        # Update the manuscript with enhanced dialogue
        updated_sections = self._enhance_dialogue_sections(
            manuscript["content"],
            dialogue_sections,
            characters,
            improvement_strategies,
            target_audience
        )
        
        # Apply the updates to the manuscript
        updated_content = self._apply_dialogue_updates(
            manuscript["content"], 
            updated_sections
        )
        
        # Store the updated manuscript
        self.document_store.update_manuscript(
            manuscript_id,
            {"content": updated_content}
        )
        
        return {
            "manuscript_id": manuscript_id,
            "message": f"Enhanced {len(updated_sections)} dialogue sections based on character profiles and target audience.",
            "dialogue_analysis": dialogue_analysis,
            "improvement_strategies": improvement_strategies
        }
    
    def _extract_dialogue_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections with significant dialogue."""
        # Split the content into paragraphs
        paragraphs = content.split("\n\n")
        
        dialogue_sections = []
        current_section = {"start": 0, "end": 0, "content": "", "dialogue_count": 0}
        
        dialogue_markers = ['"', '"', '"', ''', "'"]
        in_section = False
        section_start = 0
        dialogue_count = 0
        
        for i, para in enumerate(paragraphs):
            # Check if paragraph contains dialogue
            has_dialogue = False
            for marker in dialogue_markers:
                if marker in para:
                    has_dialogue = True
                    dialogue_count += para.count(marker) // 2  # Approximate dialogue turns
                    break
            
            # If we find dialogue and we're not in a section, start one
            if has_dialogue and not in_section:
                in_section = True
                section_start = i
                dialogue_count = para.count('"') // 2
            
            # If we're in a section and have 3+ paragraphs without dialogue, end it
            if in_section and not has_dialogue:
                no_dialogue_streak = 1
                for j in range(1, 3):
                    if i + j < len(paragraphs) and not any(m in paragraphs[i + j] for m in dialogue_markers):
                        no_dialogue_streak += 1
                
                if no_dialogue_streak >= 3:
                    # End the section
                    section_content = "\n\n".join(paragraphs[section_start:i])
                    
                    # Only save if there's significant dialogue
                    if dialogue_count >= 3:
                        dialogue_sections.append({
                            "start": section_start,
                            "end": i - 1,
                            "content": section_content,
                            "dialogue_count": dialogue_count
                        })
                    
                    in_section = False
                    dialogue_count = 0
        
        # Handle any ongoing section at the end
        if in_section and dialogue_count >= 3:
            section_content = "\n\n".join(paragraphs[section_start:])
            dialogue_sections.append({
                "start": section_start,
                "end": len(paragraphs) - 1,
                "content": section_content,
                "dialogue_count": dialogue_count
            })
        
        return dialogue_sections
    
    def _analyze_dialogue_quality(self, dialogue_sections: List[Dict[str, Any]], 
                               characters: List[Dict[str, Any]],
                               target_audience: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze the quality of dialogue in the manuscript."""
        # Sample the most dialogue-rich sections (up to 3)
        sample_sections = sorted(dialogue_sections, key=lambda x: x["dialogue_count"], reverse=True)[:3]
        sample_text = "\n\n".join([section["content"] for section in sample_sections])
        
        # Prepare character voice information
        character_voices = []
        for character in characters:
            if "voice" in character:
                character_voices.append(f"{character['name']}: {character['voice']}")
        
        character_voice_info = "\n".join(character_voices) if character_voices else "No character voice information available."
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            
            Consider whether the dialogue would resonate with this audience.
            """
        
        # Create the analysis prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a Dialogue Analysis Specialist. Analyze the dialogue quality in the following manuscript sections.
        
        Character Voice Information:
        {character_voice_info}
        
        {audience_context}
        
        Dialogue Samples:
        {dialogue_samples}
        
        Analyze the dialogue for:
        1. Authenticity and natural flow
        2. Character voice consistency 
        3. Dialog tags and attribution clarity
        4. Subtext and underlying tension
        5. Purpose (how dialogue advances plot or reveals character)
        6. Cultural or period authenticity
        7. Appeal to target audience
        
        Format your response as a detailed analysis with specific examples.
        """)
        
        # Create the chain
        chain = (
            {
                "character_voice_info": lambda _: character_voice_info,
                "dialogue_samples": lambda _: sample_text,
                "audience_context": lambda _: audience_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        analysis = chain.invoke("Analyze dialogue")
        
        return {
            "analysis": analysis,
            "sample_count": len(sample_sections),
            "total_sections": len(dialogue_sections)
        }
    
    def _determine_improvement_strategies(self, dialogue_analysis: Dict[str, Any],
                                       characters: List[Dict[str, Any]],
                                       target_audience: Optional[Dict[str, Any]] = None,
                                       research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Determine strategies for improving dialogue."""
        # Create prompt for improvement strategies
        prompt = ChatPromptTemplate.from_template("""
        You are a Dialogue Enhancement Specialist. Based on the following dialogue analysis, 
        suggest specific strategies to improve the dialogue in this manuscript.
        
        Dialogue Analysis:
        {dialogue_analysis}
        
        Character Information:
        {character_info}
        
        {audience_context}
        
        {research_context}
        
        Suggest specific strategies for:
        1. Making each character's voice more distinctive
        2. Improving natural flow and authenticity
        3. Enhancing subtext and tension
        4. Better connecting dialogue to character motivations
        5. Strengthening dialogue's contribution to plot advancement
        6. Enhancing cultural/period authenticity if relevant
        7. Making dialogue more appealing to the target audience
        
        For each strategy, provide specific examples of before/after dialogue improvements.
        """)
        
        # Prepare character info summary
        character_info = []
        for character in characters:
            char_summary = f"Name: {character['name']}\n"
            if "personality" in character:
                char_summary += f"Personality: {character['personality'][:150]}...\n"
            if "motivations" in character:
                char_summary += f"Motivations: {character['motivations'][:150]}...\n"
            if "voice" in character:
                char_summary += f"Voice: {character['voice'][:150]}...\n"
            character_info.append(char_summary)
        
        character_info_text = "\n\n".join(character_info)
        
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Language Preferences: {target_audience.get('reading_preferences', {}).get('language', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Consider how to make dialogue especially appealing to this audience.
            """
        
        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider these insights when developing the dialogue enhancement strategy.
            """
        
        # Create the chain
        chain = (
            {
                "dialogue_analysis": lambda _: dialogue_analysis["analysis"],
                "character_info": lambda _: character_info_text,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        strategies = chain.invoke("Generate improvement strategies")
        
        return {
            "strategies": strategies
        }
    
    def _enhance_dialogue_sections(self, content: str, dialogue_sections: List[Dict[str, Any]],
                               characters: List[Dict[str, Any]], improvement_strategies: Dict[str, Any],
                               target_audience: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Enhance each dialogue section based on the improvement strategies."""
        # Prepare character summary for context
        character_summaries = {}
        for character in characters:
            summary = {
                "name": character["name"],
                "voice": character.get("voice", ""),
                "personality": character.get("personality", ""),
                "motivations": character.get("motivations", "")
            }
            character_summaries[character["name"]] = summary
        
        # Prepare audience context
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Preferences: {target_audience.get('reading_preferences', {}).get('language', 'No specific preferences')}
            """
        
        # Process sections (limit to 10 for efficiency)
        processed_sections = []
        priority_sections = sorted(dialogue_sections, key=lambda x: x["dialogue_count"], reverse=True)[:10]
        
        for section in priority_sections:
            # Create enhance dialogue prompt
            prompt = ChatPromptTemplate.from_template("""
            You are a Dialogue Enhancement Specialist. Improve the dialogue in this manuscript section
            according to the provided improvement strategies.
            
            Original Section:
            {original_section}
            
            Character Information:
            {character_info}
            
            Improvement Strategies:
            {improvement_strategies}
            
            {audience_context}
            
            Enhance the dialogue in this section while maintaining the story and meaning.
            Make each character's voice more distinctive and authentic based on their profile.
            Improve subtext, tension, and connection to character motivations.
            Ensure the enhanced dialogue better advances the plot or reveals character.
            
            Return the fully enhanced section with all original content, just with improved dialogue.
            Preserve the structure and non-dialogue elements exactly as they are.
            """)
            
            # Extract character names from this section
            character_names = self._extract_character_names(section["content"], list(character_summaries.keys()))
            
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
            
            character_info_text = "\n".join(char_info) if char_info else "No specific character information available."
            
            # Create the chain
            chain = (
                {
                    "original_section": lambda _: section["content"],
                    "character_info": lambda _: character_info_text,
                    "improvement_strategies": lambda _: improvement_strategies["strategies"],
                    "audience_context": lambda _: audience_context
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            enhanced_section = chain.invoke("Enhance dialogue")
            
            processed_sections.append({
                "original": section,
                "enhanced": enhanced_section
            })
        
        return processed_sections
    
    def _apply_dialogue_updates(self, original_content: str, updated_sections: List[Dict[str, Any]]) -> str:
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
                original_section = "\n\n".join(paragraphs[start_idx:end_idx+1])
                original_sentences = self._extract_sentences(original_section)
                
                # Replace the entire section using sentence matching
                new_content = self._replace_section_by_sentences(
                    original_content, 
                    original_sentences, 
                    update["enhanced"]
                )
                
                if new_content:
                    return new_content
                
                # Fallback to paragraph replacement if sentence matching fails
            
            # Replace the paragraphs
            # Make sure we don't exceed the original content bounds
            replacement_length = min(len(enhanced_paragraphs), len(paragraphs) - start_idx)
            paragraphs[start_idx:start_idx + replacement_length] = enhanced_paragraphs[:replacement_length]
        
        # Join the paragraphs back into content
        return "\n\n".join(paragraphs)
    
    def _extract_character_names(self, text: str, known_characters: List[str]) -> List[str]:
        """Extract character names mentioned in the text."""
        # Check for exact matches of known characters
        mentioned_characters = []
        for character in known_characters:
            if character in text:
                mentioned_characters.append(character)
        
        # If no exact matches, try to find partial matches
        if not mentioned_characters:
            for character in known_characters:
                # Try first name only
                first_name = character.split()[0]
                if len(first_name) > 1 and re.search(r'\b' + re.escape(first_name) + r'\b', text):
                    mentioned_characters.append(character)
        
        return mentioned_characters
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings
        return [s for s in sentences if s]
    
    def _replace_section_by_sentences(self, original_content: str, original_sentences: List[str], 
                                  enhanced_content: str) -> Optional[str]:
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
        return original_content[:start_idx] + enhanced_content + original_content[end_idx:]
