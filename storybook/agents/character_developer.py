from __future__ import annotations
from typing import Dict, Any, Optional, List
import logging
import json
import re
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from storybook.agents.base import BaseAgent
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class CharacterDeveloper(BaseAgent):
    """Agent responsible for developing and refining characters."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)  # This is correct
        self.document_store = DocumentStore()
        # But then the LLM isn't used in the methods below

    def enhance_characters(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None, research_insights: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Identify and enhance all characters in the manuscript."""
        try:
            # Update LLM if new config provided at runtime
            if (llm_config):
                self.llm = create_llm(llm_config)

            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # Extract character names
            prompt = ChatPromptTemplate.from_template(
                """
            You are a Character Identification Specialist. Your task is to analyze the manuscript excerpt below 
            and identify all character names mentioned. Focus on characters who appear to be important to the story, 
            not just passing mentions.
            
            Manuscript: {manuscript_content}
            
            {audience_context}
            
            List each character's full name and a brief note about their apparent role.
            Format your response as a JSON list of objects with "name" and "apparent_role" keys.
            Example:
            [
              {"name": "John Smith", "apparent_role": "protagonist, detective"},
              {"name": "Mary Johnson", "apparent_role": "victim's sister"}
            ]
            """
            )

            # Add audience context if available
            audience_context = ""
            if target_audience:
                audience_context = f"""
                Target Audience Information:
                - Demographic: {target_audience.get('demographic', 'General readers')}
                - Reading Preferences: {target_audience.get('reading_preferences', {}).get('reading', 'Various preferences')}
                - Character Preferences: {target_audience.get('reading_preferences', {}).get('characters', 'Various character types')}
                
                Consider the target audience preferences when analyzing character roles and potential.
                """

            chain = (
                {
                    "manuscript_content": lambda _: manuscript.get("content", ""),
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Extract the characters
            characters_str = chain.invoke("Extract characters")
            try:
                characters_list = json.loads(characters_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse characters JSON: {characters_str}")
                # Fallback parsing
                import re

                character_matches = re.findall(r'"name":\s*"([^"]+)"', characters_str)
                characters_list = [
                    {"name": name, "apparent_role": "unknown"} for name in character_matches
                ]

            # Create detailed profiles for each character
            enhanced_characters = []
            for character in characters_list:
                profile = self.create_character_profile(
                    manuscript_id,
                    character["name"],
                    target_audience=target_audience,
                    research_insights=research_insights,
                )
                enhanced_characters.append(profile)

            return {"manuscript_id": manuscript_id, "characters": enhanced_characters}
        except Exception as e:
            logger.error(f"Error enhancing characters: {e}")
            return {"error": str(e)}

    def create_character_profile(self, manuscript_id: str, character_name: str, target_audience: Optional[Dict[str, Any]] = None, research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a detailed character profile."""
        # Get relevant manuscript parts
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")
            return {}

        # Search for character mentions
        query = f"Character {character_name}"
        character_mentions = self.document_store.get_manuscript_relevant_parts(
            manuscript_id, query
        )

        # Prepare context for the LLM
        mentions_text = "\n\n".join([doc.page_content for doc in character_mentions])

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Character Preferences: {target_audience.get('reading_preferences', {}).get('characters', 'Various character types')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Make sure the character development will resonate with this target audience.
            Consider their preferences and expectations while staying true to the manuscript's vision.
            """

        # Define the prompt
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Character Development Specialist tasked with creating a detailed character profile for a novel.
        
        Character Name: {character_name}
        
        Here are relevant excerpts from the manuscript where this character appears:
        
        {character_mentions}
        
        {audience_context}
        
        Based on these excerpts and your expertise, create a detailed character profile with the following sections:
        
        1. Physical Description: Appearance, mannerisms, distinctive features
        2. Background & History: Family, upbringing, significant past events
        3. Personality: Core traits, values, flaws, strengths
        4. Motivations: What drives this character? What are their goals?
        5. Relationships: How they connect with other characters
        6. Character Arc: How they might develop through the story
        7. Voice: Distinctive speech patterns, vocabulary, expressions
        8. Target Audience Appeal: How this character might resonate with the target readers
        
        Make reasonable inferences where information is missing, staying true to what's implied in the text.
        If you notice inconsistencies in how the character is portrayed, note them for the author's attention.
        """
        )

        # Create the chain
        chain = (
            {
                "character_name": lambda _: character_name,
                "character_mentions": lambda _: mentions_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        profile_text = chain.invoke(character_name)

        # Parse the profile into sections
        sections = [
            "Physical Description",
            "Background & History",
            "Personality",
            "Motivations",
            "Relationships",
            "Character Arc",
            "Voice",
            "Target Audience Appeal",
        ]

        profile = {"name": character_name, "full_profile": profile_text}

        # Extract sections
        for i in range(len(sections)):
            start_marker = sections[i]
            end_marker = sections[i + 1] if i < len(sections) - 1 else None

            start_idx = profile_text.find(start_marker)
            if start_idx == -1:
                continue

            start_idx += len(start_marker)
            end_idx = (
                profile_text.find(end_marker, start_idx)
                if end_marker
                else len(profile_text)
            )

            content = profile_text[start_idx:end_idx].strip()
            key = start_marker.lower().replace(" & ", "_").replace(" ", "_")
            profile[key] = content

        # Store the character profile
        character_id = self.document_store.store_character_details(
            manuscript_id, character_name, profile
        )

        # Add the ID to the profile
        profile["id"] = character_id

        return profile

    def process_manuscript(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]], research_insights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process manuscript for character development."""
        try:
            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            characters = self._extract_characters(manuscript["content"])
            character_profiles = []

            for character in characters:
                profile = self.create_character_profile(
                    manuscript_id,
                    character,
                    target_audience,
                    research_insights
                )
                character_profiles.append(profile)

            return {
                "manuscript_id": manuscript_id,
                "characters": character_profiles,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in character development: {str(e)}")
            return self.handle_error(e)
