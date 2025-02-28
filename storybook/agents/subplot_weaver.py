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


class SubplotWeaver(BaseAgent):
    """Agent responsible for subplot integration and development."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()

    def weave_subplots(
        self,
        manuscript_id: str,
        characters: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Identify and integrate subplots in the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}

        # Identify existing subplots
        existing_subplots = self._identify_subplots(
            manuscript["content"], characters, target_audience
        )

        # Analyze subplot potential
        subplot_potential = self._analyze_subplot_potential(
            manuscript["content"],
            characters,
            existing_subplots,
            target_audience,
            research_insights,
        )

        # Develop and integrate subplots
        developed_subplots = self._develop_subplots(
            manuscript_id,
            existing_subplots,
            subplot_potential,
            characters,
            target_audience,
            research_insights,
        )

        # Integrate subplots into the manuscript
        updated_content = self._integrate_subplots(
            manuscript["content"], developed_subplots, characters, target_audience
        )

        # Store the updated manuscript
        self.document_store.update_manuscript(
            manuscript_id, {"content": updated_content}
        )

        return {
            "manuscript_id": manuscript_id,
            "existing_subplots": existing_subplots,
            "subplot_potential": subplot_potential,
            "developed_subplots": developed_subplots,
            "message": f"Identified {len(existing_subplots)} existing subplots and developed {len(developed_subplots)} subplots.",
        }

    def _identify_subplots(
        self,
        content: str,
        characters: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: 
        """Identify existing subplots in the manuscript."""
        # Prepare character information
        character_names = [character["name"] for character in characters]
        character_text = ", ".join(character_names)

        # Create sampling of the manuscript
        sample_size = min(8000, len(content) // 3)
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
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Consider which subplots would be most engaging for this audience.
            """

        # Create prompt for subplot identification
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Subplot Analysis Specialist. Identify the existing subplots in this manuscript.
        
        Manuscript Sample:
        {manuscript_sample}
        
        Characters: {characters}
        
        {audience_context}
        
        A subplot is a secondary strand of the plot that supports the main plot. For each subplot you identify, provide:
        
        1. Title: A descriptive title for the subplot
        2. Characters Involved: Which characters are central to this subplot
        3. Description: Brief summary of what this subplot entails
        4. Status: How developed this subplot currently is (underdeveloped, partially developed, well-developed)
        5. Connection to Main Plot: How this subplot relates to the main storyline
        
        Format your response as a JSON list of objects with the above keys.
        Focus on actual existing subplots, not potential ones.
        """
        )

        # Create the chain
        chain = (
            {
                "manuscript_sample": lambda _: sample,
                "characters": lambda _: character_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        subplots_str = chain.invoke("Identify subplots")

        # Parse the subplots

        try:
            subplots_list = json.loads(subplots_str)
            if isinstance(subplots_list, list):
                return subplots_list
            return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse subplots JSON: {subplots_str}")
            # Try a more lenient approach to extract structured data

            subplots = []

            # Try to find subplot sections
            subplot_blocks = re.findall(
                r"(\d+\.\s+Title:.*?)(?=\d+\.\s+Title:|$)", subplots_str, re.DOTALL
            )

            for block in subplot_blocks:
                subplot = {}

                # Extract title
                title_match = re.search(r"Title:\s*(.*?)(?=\n|$)", block)
                if title_match:
                    subplot["title"] = title_match.group(1).strip()

                # Extract characters
                chars_match = re.search(r"Characters Involved:\s*(.*?)(?=\n|$)", block)
                if chars_match:
                    subplot["characters_involved"] = chars_match.group(1).strip()

                # Extract description
                desc_match = re.search(
                    r"Description:\s*(.*?)(?=\n\d|$)", block, re.DOTALL
                )
                if desc_match:
                    subplot["description"] = desc_match.group(1).strip()

                # Extract status
                status_match = re.search(r"Status:\s*(.*?)(?=\n|$)", block)
                if status_match:
                    subplot["status"] = status_match.group(1).strip()

                # Extract connection
                conn_match = re.search(
                    r"Connection to Main Plot:\s*(.*?)(?=\n|$)", block, re.DOTALL
                )
                if conn_match:
                    subplot["connection_to_main_plot"] = conn_match.group(1).strip()

                if subplot.get("title"):
                    subplots.append(subplot)

            return subplots

    def _analyze_subplot_potential(
        self,
        content: str,
        characters: List[Dict[str, Any]],
        existing_subplots: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze potential for new or enhanced subplots."""
        # Format character information
        character_summaries = []
        for character in characters[:5]:  # Limit to top 5 characters
            summary = f"Name: {character['name']}\n"
            if "motivations" in character:
                summary += f"Motivations: {character['motivations'][:150]}...\n"
            if "personality" in character:
                summary += f"Personality: {character['personality'][:150]}...\n"
            character_summaries.append(summary)

        character_text = "\n\n".join(character_summaries)

        # Format existing subplots
        existing_subplot_summaries = []
        for subplot in existing_subplots:
            summary = f"Title: {subplot.get('title', 'Untitled')}\n"
            summary += f"Characters: {subplot.get('characters_involved', 'Unknown')}\n"
            summary += f"Status: {subplot.get('status', 'Unknown')}\n"
            existing_subplot_summaries.append(summary)

        existing_subplot_text = (
            "\n\n".join(existing_subplot_summaries)
            if existing_subplot_summaries
            else "No existing subplots identified."
        )

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Consider subplot elements that would particularly resonate with this audience.
            """

        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider how subplots could align with current market trends and reader preferences.
            """

        # Create prompt for subplot potential analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Subplot Development Specialist. Analyze the potential for new or enhanced subplots in this manuscript.
        
        Character Information:
        {character_info}
        
        Existing Subplots:
        {existing_subplots}
        
        {audience_context}
        
        {research_context}
        
        Provide a detailed analysis of:
        
        1. Potential relationships or conflicts between characters that could be developed into subplots
        2. Thematic elements that could be explored through new subplots
        3. Opportunities to enhance existing subplots
        4. Character arcs that could benefit from subplot development
        5. Subplot ideas that would appeal to the target audience
        
        For each opportunity, explain how it would strengthen the overall narrative and character development.
        """
        )

        # Create the chain
        chain = (
            {
                "character_info": lambda _: character_text,
                "existing_subplots": lambda _: existing_subplot_text,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        potential_analysis = chain.invoke("Analyze subplot potential")

        return {
            "potential_analysis": potential_analysis,
            "existing_subplot_count": len(existing_subplots),
        }

    def _develop_subplots(
        self,
        manuscript_id: str,
        existing_subplots: List[Dict[str, Any]],
        subplot_potential: Dict[str, Any],
        characters: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: 
        """Develop and enhance subplots based on the analysis."""
        # Format characters for reference
        character_dict = {character["name"]: character for character in characters}

        # Categorize existing subplots
        underdeveloped_subplots = []
        developed_subplots = []

        for subplot in existing_subplots:
            status = subplot.get("status", "").lower()
            if "underdeveloped" in status or "partially" in status:
                underdeveloped_subplots.append(subplot)
            else:
                developed_subplots.append(subplot)

        # Determine how many new subplots to create (1-2)
        num_new_subplots = (
            min(2, 3 - len(existing_subplots)) if len(existing_subplots) < 3 else 0
        )

        all_developed_subplots = []

        # Develop underdeveloped existing subplots
        for subplot in underdeveloped_subplots:
            developed_subplot = self._enhance_existing_subplot(
                manuscript_id,
                subplot,
                character_dict,
                target_audience,
                research_insights,
            )
            all_developed_subplots.append(developed_subplot)

        # Add already developed subplots
        for subplot in developed_subplots:
            # Store in the database
            subplot_id = self.document_store.store_subplot(
                manuscript_id, subplot.get("title", "Untitled Subplot"), subplot
            )

            # Add the ID
            subplot["id"] = subplot_id
            all_developed_subplots.append(subplot)

        # Create new subplots if needed
        if num_new_subplots > 0:
            new_subplots = self._create_new_subplots(
                manuscript_id,
                num_new_subplots,
                existing_subplots,
                subplot_potential,
                character_dict,
                target_audience,
                research_insights,
            )
            all_developed_subplots.extend(new_subplots)

        return all_developed_subplots

    def _enhance_existing_subplot(
        self,
        manuscript_id: str,
        subplot: Dict[str, Any],
        character_dict: Dict[str, Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: 
        """Enhance an existing underdeveloped subplot."""
        subplot_title = subplot.get("title", "Untitled Subplot")

        # Get relevant character information
        character_info = []
        characters_involved = subplot.get("characters_involved", "")

        for char_name, char_data in character_dict.items():
            if char_name in characters_involved:
                info = f"Character: {char_name}\n"
                if "motivations" in char_data:
                    info += f"Motivations: {char_data['motivations'][:200]}\n"
                if "personality" in char_data:
                    info += f"Personality: {char_data['personality'][:200]}\n"
                character_info.append(info)

        character_text = "\n\n".join(character_info)

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Consider how to enhance this subplot to appeal to this target audience.
            """

        # Create prompt for subplot enhancement
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Subplot Development Specialist. Enhance this existing subplot to make it more compelling and fully developed.
        
        Subplot Information:
        Title: {subplot_title}
        Current Description: {subplot_description}
        Status: {subplot_status}
        Connection to Main Plot: {subplot_connection}
        
        Character Information:
        {character_info}
        
        {audience_context}
        
        Develop this subplot by creating:
        
        1. Enhanced Subplot Arc: A more detailed progression with clear beginning, middle, and end
        2. Key Scenes: 3-5 specific scenes that would develop this subplot
        3. Character Development: How this subplot contributes to character growth
        4. Thematic Enhancement: How this subplot reinforces key themes
        5. Integration Points: Specific ways to weave this subplot into the main narrative
        6. Resolution: How this subplot should be resolved
        
        Make the subplot emotionally resonant and meaningful to the overall story.
        """
        )

        # Create the chain
        chain = (
            {
                "subplot_title": lambda _: subplot_title,
                "subplot_description": lambda _: subplot.get("description", ""),
                "subplot_status": lambda _: subplot.get("status", ""),
                "subplot_connection": lambda _: subplot.get(
                    "connection_to_main_plot", ""
                ),
                "character_info": lambda _: character_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        enhanced_subplot = chain.invoke(f"Enhance subplot: {subplot_title}")

        # Parse the enhanced subplot into sections
        sections = [
            "Enhanced Subplot Arc",
            "Key Scenes",
            "Character Development",
            "Thematic Enhancement",
            "Integration Points",
            "Resolution",
        ]

        result = {
            "title": subplot_title,
            "original_description": subplot.get("description", ""),
            "characters_involved": subplot.get("characters_involved", ""),
            "connection_to_main_plot": subplot.get("connection_to_main_plot", ""),
            "full_development": enhanced_subplot,
        }

        # Extract sections
        for i in range(len(sections)):
            start_marker = sections[i]
            end_marker = sections[i + 1] if i < len(sections) - 1 else None

            start_idx = enhanced_subplot.find(start_marker)
            if start_idx == -1:
                continue

            start_idx += len(start_marker)
            end_idx = (
                enhanced_subplot.find(end_marker, start_idx)
                if end_marker
                else len(enhanced_subplot)
            )

            content = enhanced_subplot[start_idx:end_idx].strip()
            key = start_marker.lower().replace(" ", "_")
            result[key] = content

        # Store in the database
        subplot_id = self.document_store.store_subplot(
            manuscript_id, subplot_title, result
        )

        # Add the ID
        result["id"] = subplot_id

        return result

    def _create_new_subplots(
        self,
        manuscript_id: str,
        num_subplots: int,
        existing_subplots: List[Dict[str, Any]],
        subplot_potential: Dict[str, Any],
        character_dict: Dict[str, Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: 
        """Create new subplots based on the potential analysis."""
        # Format existing subplot information
        existing_subplot_titles = [
            subplot.get("title", "Untitled") for subplot in existing_subplots
        ]
        existing_subplot_text = (
            ", ".join(existing_subplot_titles) if existing_subplot_titles else "None"
        )

        # Format character information
        character_info = []
        for char_name, char_data in list(character_dict.items())[ 
            :5
        ]:  # Limit to top 5 characters
            info = f"Character: {char_name}\n"
            if "motivations" in char_data:
                info += f"Motivations: {char_data['motivations'][:150]}\n"
            if "personality" in char_data:
                info += f"Personality: {char_data['personality'][:150]}\n"
            character_info.append(info)

        character_text = "\n\n".join(character_info)

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Research Insights:
            {research_insights.get('market_analysis_summary', '') if research_insights else ''}
            
            Create subplots that would particularly resonate with this target audience and align with market trends.
            """

        # Create prompt for new subplot creation
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Subplot Creation Specialist. Create {num_subplots} new compelling subplots for this manuscript.
        
        Existing Subplots: {existing_subplots}
        
        Subplot Potential Analysis:
        {potential_analysis}
        
        Character Information:
        {character_info}
        
        {audience_context}
        
        For each new subplot, provide:
        
        1. Title: A descriptive title
        2. Characters Involved: Which characters are central to this subplot
        3. Subplot Arc: A detailed progression with beginning, middle, and end
        4. Key Scenes: 3-5 specific scenes that would develop this subplot
        5. Character Development: How this subplot contributes to character growth
        6. Thematic Value: How this subplot reinforces or introduces themes
        7. Integration Points: Specific ways to weave this subplot into the main narrative
        8. Resolution: How this subplot should be resolved
        
        Create subplots that enhance the overall narrative, deepen character development, and provide meaningful emotional arcs.
        Make sure the subplots don't distract from the main plot but rather complement and strengthen it.
        """
        )

        # Create the chain
        chain = (
            {
                "num_subplots": lambda _: num_subplots,
                "existing_subplots": lambda _: existing_subplot_text,
                "potential_analysis": lambda _: subplot_potential.get(
                    "potential_analysis", ""
                ),
                "character_info": lambda _: character_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        new_subplots_text = chain.invoke("Create new subplots")

        # Parse the new subplots
        # This is tricky as the format might vary, so we'll try to identify subplot sections
        # Try to find subplot sections (numbered or with "Subplot 1:" style headers)
        subplot_blocks = re.split(
            r"(?:Subplot\s+\d+:|New Subplot\s+\d+:|#\d+\s+Subplot|\d+\.\s+Subplot)",
            new_subplots_text,
        )
        # Remove any empty first element from the split
        if subplot_blocks and not subplot_blocks[0].strip():
            subplot_blocks = subplot_blocks[1:]

        # If we couldn't identify clear blocks, try another approach
        if len(subplot_blocks) != num_subplots:
            # Try splitting by "Title:" occurrences, assuming each subplot starts with a title
            subplot_blocks = re.split(r"(?:\n|^)Title:", new_subplots_text)
            # Remove any empty first element
            if subplot_blocks and not subplot_blocks[0].strip():
                subplot_blocks = subplot_blocks[1:]
            # Add back the "Title:" that was removed by the split
            subplot_blocks = ["Title:" + block for block in subplot_blocks]

        # Process each subplot block
        new_subplots = []
        for i, block in enumerate(subplot_blocks[:num_subplots]):
            subplot = self._parse_new_subplot(block, manuscript_id)
            if subplot:
                new_subplots.append(subplot)

        return new_subplots

    def _parse_new_subplot(
        self, subplot_text: str, manuscript_id: str
    ) -> Optional[Dict[str, Any]]: 
        """Parse a new subplot from text."""
        # Define the sections we expect to find
        sections = [
            "Title",
            "Characters Involved",
            "Subplot Arc",
            "Key Scenes",
            "Character Development",
            "Thematic Value",
            "Integration Points",
            "Resolution",
        ]

        result = {"full_development": subplot_text}

        # Extract title first - it's most important
        title_match = re.search(r"Title:?\s*(.*?)(?=\n|$)", subplot_text)
        if title_match:
            result["title"] = title_match.group(1).strip()
        else:
            # If no title, create a generic one
            result["title"] = f"New Subplot {manuscript_id[-4:]}"

        # Extract characters involved
        chars_match = re.search(
            r"Characters Involved:?\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", subplot_text, re.DOTALL
        )
        if chars_match:
            result["characters_involved"] = chars_match.group(1).strip()

        # Try to extract other sections
        for section in sections[2:]:  # Skip Title and Characters which we handled above
            pattern = re.compile(f"{section}:?\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", re.DOTALL)
            match = pattern.search(subplot_text)
            if match:
                key = section.lower().replace(" ", "_")
                result[key] = match.group(1).strip()

        # Store in the database
        subplot_id = self.document_store.store_subplot(
            manuscript_id, result["title"], result
        )

        # Add the ID
        result["id"] = subplot_id

        return result

    def _integrate_subplots(
        self,
        content: str,
        developed_subplots: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> str: 
        """Integrate subplot elements into the manuscript."""
        # We'll focus on integrating a few crucial scenes from each subplot
        # rather than trying to completely rewrite the manuscript

        # Sort subplots by importance - we'll prioritize new ones first
        priority_subplots = sorted(
            developed_subplots, key=lambda x: 0 if "integration_points" in x else 1
        )

        # Limit to top 3 subplots to avoid too many changes
        subplots_to_integrate = priority_subplots[: min(3, len(priority_subplots))]

        updated_content = content

        for subplot in subplots_to_integrate:
            # Skip if no integration points specified
            if "integration_points" not in subplot:
                continue

            # Extract information about where to integrate
            integration_points = subplot["integration_points"]
            key_scenes = subplot.get("key_scenes", "")

            # Create prompt for integration
            prompt = ChatPromptTemplate.from_template(
                """
            You are a Subplot Integration Specialist. Create a scene that integrates this subplot into the manuscript.
            
            Subplot Title: {subplot_title}
            
            Subplot Characters: {subplot_characters}
            
            Key Scenes to Develop:
            {key_scenes}
            
            Integration Points:
            {integration_points}
            
            {audience_context}
            
            Create a single scene (1-3 paragraphs) that could be inserted into the manuscript to develop this subplot.
            The scene should:
            1. Feel natural and consistent with the existing manuscript style
            2. Focus on a key moment from the subplot
            3. Involve the characters central to the subplot
            4. Connect to the main plot in a meaningful way
            
            Write the scene in the manuscript's existing style and voice.
            """
            )

            # Add audience context if available
            audience_context = ""
            if target_audience:
                audience_context = f"""
                Target Audience: {target_audience.get('demographic', 'General readers')}
                Ensure the scene will resonate with this audience.
                """

            # Create the chain
            chain = (
                {
                    "subplot_title": lambda _: subplot.get("title", "Untitled Subplot"),
                    "subplot_characters": lambda _: subplot.get(
                        "characters_involved", ""
                    ),
                    "key_scenes": lambda _: key_scenes,
                    "integration_points": lambda _: integration_points,
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            new_scene = chain.invoke(
                f"Create integration scene for {subplot.get('title', 'subplot')}"
            )

            # Find a suitable insertion point
            # This is tricky and depends on the manuscript structure
            # For this implementation, we'll insert near the middle of the manuscript
            # A more sophisticated approach would analyze the content to find natural breaks

            # Calculate insertion point (adjusted for each subplot to spread them out)
            insertion_idx = len(updated_content) // 2

            # For each subplot, shift the insertion point
            subplot_idx = subplots_to_integrate.index(subplot)
            offset = (subplot_idx - len(subplots_to_integrate) // 2) * (
                len(updated_content) // 8
            )
            insertion_idx += offset

            # Find the nearest paragraph break
            paragraph_break = updated_content.rfind("\n\n", 0, insertion_idx)
            if paragraph_break == -1:
                paragraph_break = updated_content.find("\n\n", insertion_idx)

            if paragraph_break != -1:
                # Insert the new scene
                updated_content = (
                    updated_content[:paragraph_break]
                    + "\n\n"
                    + new_scene
                    + "\n\n"
                    + updated_content[paragraph_break:]
                )

        return updated_content
