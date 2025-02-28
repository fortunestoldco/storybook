from typing import Dict, List, Any, Optional
import logging
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document  # Add for MongoDB operations

from storybook.config import get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class WorldBuilder:
    """Agent responsible for world-building and enhancing settings."""

    def __init__(self):
        self.llm = get_llm(temperature=0.7, use_replicate=True)
        self.document_store = DocumentStore()

    def build_world(
        self,
        manuscript_id: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Identify and enhance settings and world-building elements."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}

        # Extract settings and locations
        settings = self._extract_settings(
            manuscript["content"], target_audience, research_insights
        )

        # Develop each setting
        enhanced_settings = []
        for setting in settings:
            setting_profile = self._develop_setting(
                manuscript_id,
                setting,
                manuscript["content"],
                target_audience,
                research_insights,
            )
            enhanced_settings.append(setting_profile)

        # Analyze world consistency
        world_consistency = self._analyze_world_consistency(
            enhanced_settings, manuscript["content"], target_audience
        )

        # Update the manuscript with enhanced world descriptions
        updated_content = self._enhance_setting_descriptions(
            manuscript["content"], enhanced_settings, target_audience
        )

        # Store the updated manuscript
        self.document_store.update_manuscript(
            manuscript_id, {"content": updated_content}
        )

        return {
            "manuscript_id": manuscript_id,
            "settings": enhanced_settings,
            "world_consistency": world_consistency,
            "message": f"Enhanced {len(enhanced_settings)} settings and world-building elements.",
        }

    def _extract_settings(
        self,
        content: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract settings and locations from the manuscript."""
        # Sample representative portions of the manuscript
        sample_size = min(6000, len(content) // 4)
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
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Consider what settings would resonate with this audience.
            """

        # Create prompt for settings extraction
        prompt = ChatPromptTemplate.from_template(
            """
        You are a World-Building Specialist. Identify all settings and locations in the manuscript samples.
        
        Manuscript Samples:
        {manuscript_sample}
        
        {audience_context}
        
        For each setting or location, provide:
        1. Name/identifier
        2. Brief description
        3. Apparent importance to the story (primary, secondary, tertiary)
        4. Any notable features or atmosphere described
        
        Focus on physical locations and settings, not abstract concepts.
        Format your response as a JSON list of objects with keys: "name", "description", "importance", and "features".
        """
        )

        # Create the chain
        chain = (
            {
                "manuscript_sample": lambda _: sample,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        settings_str = chain.invoke("Extract settings")

        # Parse the settings
        try:
            settings_list = json.loads(settings_str)
            if isinstance(settings_list, list):
                return settings_list
            return []
        except json.JSONDecodeError:
            # Fallback - try to extract structured content
            logger.error(f"Failed to parse settings JSON: {settings_str}")

            # Extract setting entries using regex
            settings = []
            setting_blocks = re.findall(r'{\s*"name":[^}]+}', settings_str)

            for block in setting_blocks:
                try:
                    setting = json.loads(block)
                    settings.append(setting)
                except:
                    continue

            return settings

    def _develop_setting(
        self,
        manuscript_id: str,
        setting: Dict[str, Any],
        content: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Develop a detailed profile for a setting."""
        setting_name = setting.get("name", "")

        # Find mentions of this setting
        query = f"setting {setting_name} location place"
        setting_mentions = self.document_store.get_manuscript_relevant_parts(
            manuscript_id, query
        )

        # Combine mentions
        mentions_text = "\n\n".join([doc.page_content for doc in setting_mentions])

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Research Insights:
            {research_insights.get('market_analysis_summary', '') if research_insights else ''}
            
            Consider these insights when developing the setting to ensure it resonates with the target audience.
            """

        # Create prompt for setting development
        prompt = ChatPromptTemplate.from_template(
            """
        You are a World-Building Specialist developing a detailed setting profile for a novel.
        
        Setting Name: {setting_name}
        
        Initial Description:
        {initial_description}
        
        Relevant Manuscript Excerpts:
        {setting_mentions}
        
        {audience_context}
        
        Create a detailed profile for this setting with the following elements:
        
        1. Physical Description: Detailed visual and sensory attributes
        2. History & Background: Origins and development of this location
        3. Cultural Significance: How this place affects characters and society
        4. Atmosphere & Mood: Emotional quality and psychological impact
        5. Symbolic Meaning: What this setting represents thematically
        6. Points of Interest: Notable features or sub-locations
        7. Potential Development: How this setting could be enhanced or expanded
        8. Target Audience Appeal: How this setting might resonate with readers
        
        Make reasonable inferences based on the manuscript excerpts, staying true to the author's vision.
        Provide specific suggestions for enriching this setting to deepen immersion.
        """
        )

        # Create the chain
        chain = (
            {
                "setting_name": lambda _: setting_name,
                "initial_description": lambda _: setting.get("description", ""),
                "setting_mentions": lambda _: mentions_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        profile_text = chain.invoke(setting_name)

        # Parse the profile into sections
        sections = [
            "Physical Description",
            "History & Background",
            "Cultural Significance",
            "Atmosphere & Mood",
            "Symbolic Meaning",
            "Points of Interest",
            "Potential Development",
            "Target Audience Appeal",
        ]

        profile = {
            "name": setting_name,
            "original_description": setting.get("description", ""),
            "importance": setting.get("importance", ""),
            "features": setting.get("features", ""),
            "full_profile": profile_text,
        }

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

        # Store the setting profile
        setting_id = self.document_store.store_world_details(
            manuscript_id, setting_name, profile
        )

        # Add the ID to the profile
        profile["id"] = setting_id

        return profile

    def _analyze_world_consistency(
        self,
        settings: List[Dict[str, Any]],
        content: str,
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze the consistency of the world-building elements."""
        # Prepare settings summary
        settings_summary = []
        for setting in settings:
            summary = f"Setting: {setting['name']}\n"
            if "physical_description" in setting:
                summary += f"Description: {setting['physical_description'][:200]}...\n"
            if "history_&_background" in setting:
                summary += f"History: {setting['history_&_background'][:200]}...\n"
            settings_summary.append(summary)

        settings_text = "\n\n".join(settings_summary)

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Consider whether the world-building would appeal to this audience.
            """

        # Create prompt for consistency analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a World-Building Consistency Specialist. Analyze the consistency and coherence of the world-building
        in this manuscript based on the identified settings.
        
        Settings Information:
        {settings_info}
        
        {audience_context}
        
        Provide a detailed analysis of:
        1. Overall world consistency and coherence
        2. Potential contradictions or inconsistencies
        3. Underdeveloped aspects that need more attention
        4. Strengths in the world-building
        5. Opportunities to improve immersion and believability
        6. Suggestions for target audience appeal
        
        Format your response as a comprehensive report with specific examples.
        """
        )

        # Create the chain
        chain = (
            {
                "settings_info": lambda _: settings_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        analysis = chain.invoke("Analyze world consistency")

        return {"consistency_analysis": analysis, "setting_count": len(settings)}

    def _enhance_setting_descriptions(
        self,
        content: str,
        settings: List[Dict[str, Any]],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enhance setting descriptions in the manuscript."""
        # Sort settings by importance
        priority_settings = sorted(
            settings,
            key=lambda x: (
                0
                if x.get("importance", "").lower() == "primary"
                else (1 if x.get("importance", "").lower() == "secondary" else 2)
            ),
        )

        # Process a limited number of settings to avoid too many changes
        settings_to_enhance = priority_settings[: min(5, len(priority_settings))]

        updated_content = content

        for setting in settings_to_enhance:
            setting_name = setting["name"]

            # Find first significant mention of this setting
            pattern = re.compile(
                r"(\.\s+|^)([^.!?]*\b" + re.escape(setting_name) + r"\b[^.!?]*[.!?])"
            )
            match = pattern.search(updated_content)

            if not match:
                continue

            # Extract the sentence containing the setting
            original_sentence = match.group(2)

            # Create prompt for enhancing description
            prompt = ChatPromptTemplate.from_template(
                """
            You are a World-Building Enhancement Specialist. Enhance this setting description based on the detailed profile.
            
            Setting Name: {setting_name}
            
            Original Sentence: {original_sentence}
            
            Detailed Setting Profile:
            {setting_profile}
            
            {audience_context}
            
            Create an enhanced description that:
            1. Maintains the same basic information as the original
            2. Incorporates vivid sensory details from the detailed profile
            3. Adds atmospheric elements suggested in the profile
            4. Keeps the voice and style consistent with the rest of the manuscript
            
            The enhanced description should be a paragraph of 2-3 sentences that could directly replace the original sentence.
            Make sure it flows naturally with the surrounding text and doesn't feel artificially inserted.
            """
            )

            # Prepare audience context
            audience_context = ""
            if target_audience:
                audience_context = f"""
                Target Audience: {target_audience.get('demographic', 'General readers')}
                Ensure the enhanced description appeals to this audience.
                """

            # Prepare setting profile summary
            profile_parts = []
            if "physical_description" in setting:
                profile_parts.append(
                    f"Physical Description: {setting['physical_description']}"
                )
            if "atmosphere_&_mood" in setting:
                profile_parts.append(
                    f"Atmosphere & Mood: {setting['atmosphere_&_mood']}"
                )
            if "symbolic_meaning" in setting:
                profile_parts.append(f"Symbolic Meaning: {setting['symbolic_meaning']}")

            profile_text = "\n\n".join(profile_parts)

            # Create the chain
            chain = (
                {
                    "setting_name": lambda _: setting_name,
                    "original_sentence": lambda _: original_sentence,
                    "setting_profile": lambda _: profile_text,
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            enhanced_description = chain.invoke(f"Enhance {setting_name}")

            # Replace the original sentence with the enhanced description
            updated_content = updated_content.replace(
                original_sentence, enhanced_description, 1
            )

        return updated_content
