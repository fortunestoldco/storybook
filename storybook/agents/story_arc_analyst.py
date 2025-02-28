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

logger = logging.getLogger(__name__)

class StoryArcAnalyst(BaseAgent):
    """Agent responsible for analyzing and improving story arcs."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()

    def analyze_story_arc(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None, research_insights: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if llm_config:
                self.llm = create_llm(llm_config)

            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # Implementation here
            arc_analysis = self._analyze_arc_components(manuscript["content"], target_audience)
            pacing_analysis = self._analyze_pacing(manuscript["content"], arc_analysis, target_audience)

            return {
                "manuscript_id": manuscript_id,
                "arc_analysis": arc_analysis,
                "pacing_analysis": pacing_analysis
            }
        except Exception as e:
            logger.error(f"Error in analyze_story_arc: {str(e)}")
            return self.handle_error(e)

    def _extract_arc_metrics(self, analysis_text: str) -> Dict[str, str]:
        """Extract metrics from analysis text using regex."""
        metrics = {}
        
        # Extract pacing balance
        balance_match = re.search(r"Pacing Balance:\s*(.*?)(?=\n|$)", analysis_text)
        metrics["pacing_balance"] = balance_match.group(1).strip() if balance_match else ""
        
        # Extract tension curve
        tension_match = re.search(r"Tension Curve:\s*(.*?)(?=\n|$)", analysis_text)
        metrics["tension_curve"] = tension_match.group(1).strip() if tension_match else ""
        
        return metrics

    def _analyze_arc_components(self, content: str, target_audience: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze major components of the story arc."""
        try:
            return {
                "setup": self._analyze_section(content[:len(content)//4]),
                "rising_action": self._analyze_section(content[len(content)//4:len(content)//2]),
                "climax": self._analyze_section(content[len(content)//2:3*len(content)//4]),
                "resolution": self._analyze_section(content[3*len(content)//4:])
            }
        except Exception as e:
            logger.error(f"Error analyzing arc components: {str(e)}")
            return {}

    def _analyze_section(self, section: str) -> Dict[str, Any]:
        """Analyze a specific section of the story."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """Analyze this section of the story and provide:
                1. Key events
                2. Emotional intensity (1-10)
                3. Pacing (slow/medium/fast)
                4. Character development
                5. Plot advancement

                Section:
                {section}
                """
            )
            
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"section": section[:2000]})  # Analyze first 2000 chars
            
            return self._parse_section_analysis(result)
            
        except Exception as e:
            logger.error(f"Error analyzing section: {str(e)}")
            return {}

    def _parse_section_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse the section analysis into structured data."""
        return {
            "key_events": self._extract_key_events(analysis),
            "emotional_intensity": self._extract_intensity(analysis),
            "pacing": self._extract_pacing(analysis),
            "character_development": self._extract_character_dev(analysis),
            "plot_advancement": self._extract_plot_advancement(analysis)
        }

    def _generate_improvement_plan(
        self,
        structure_analysis: Dict[str, Any],
        character_arcs: List[Dict[str, Any]],
        pacing_analysis: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive plan for improving story arcs and structure."""
        # Format the analyses
        structure_summary = f"""
        Overall Structure: {structure_analysis.get('overall_structure', '')}
        Key Plot Points: {structure_analysis.get('key_plot_points', '')}
        Inciting Incident: {structure_analysis.get('inciting_incident', '')}
        Midpoint: {structure_analysis.get('midpoint', '')}
        Climax: {structure_analysis.get('climax', '')}
        Resolution: {structure_analysis.get('resolution', '')}
        Structural Weaknesses: {structure_analysis.get('weaknesses', '')}
        """

        # Summarize character arcs
        character_summary = []
        for arc in character_arcs:
            summary = f"Character: {arc.get('character_name', '')}\n"
            summary += f"Arc Type: {arc.get('arc_type', '')}\n"
            summary += f"Effectiveness: {arc.get('effectiveness', '')}\n"
            summary += f"Improvement Opportunities: {arc.get('improvement_opportunities', '')}\n"
            character_summary.append(summary)

        character_summary_text = "\n\n".join(character_summary)

        # Summarize pacing
        pacing_summary = f"""
        Overall Pacing: {pacing_analysis.get('overall_pacing', '')}
        Pacing Balance: {pacing_analysis.get('pacing_balance', '')}
        Pacing Issues: {pacing_analysis.get('issues', '')}
        """

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Reading Preferences: {target_audience.get('reading_preferences', {}).get('reading', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Tailor your improvement recommendations to ensure they will resonate with this audience.
            """

        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider these market trends when recommending story arc improvements.
            """

        # Create prompt for improvement plan
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Story Arc Improvement Specialist. Create a comprehensive plan to improve the story arcs
        and structure of this manuscript based on the analyses provided.
        
        Structure Analysis:
        {structure_summary}
        
        Character Arc Analysis:
        {character_summary}
        
        Pacing Analysis:
        {pacing_summary}
        
        {audience_context}
        
        {research_context}
        
        Create a detailed improvement plan addressing:
        
        1. Structure Improvements: How to strengthen the overall narrative structure
        2. Plot Enhancement: How to make the plot more compelling and cohesive
        3. Character Arc Refinements: Specific ways to strengthen character journeys
        4. Pacing Adjustments: How to improve rhythm and tension
        5. Key Scene Additions or Modifications: Specific scenes to add, remove, or change
        6. Thematic Reinforcement: How to better communicate core themes
        7. Target Audience Appeal: How to make the story more appealing to the target readers
        
        For each recommendation, provide:
        - A clear explanation of the issue
        - A specific suggestion for improvement
        - An example of how to implement the change
        
        Prioritize improvements that will have the greatest impact on reader engagement.
        """
        )

        # Create the chain
        chain = (
            {
                "structure_summary": lambda _: structure_summary,
                "character_summary": lambda _: character_summary_text,
                "pacing_summary": lambda _: pacing_summary,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        improvement_plan = chain.invoke("Generate improvement plan")

        # Parse the plan into sections

        # Extract key sections
        structure_improvements_match = re.search(
            r"Structure Improvements:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )
        plot_match = re.search(
            r"Plot Enhancement:?\s*(.*?)(?=\n\n|\n\d\.|\Z)", improvement_plan, re.DOTALL
        )
        character_match = re.search(
            r"Character Arc Refinements:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )
        pacing_match = re.search(
            r"Pacing Adjustments:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )
        scenes_match = re.search(
            r"Key Scene Additions or Modifications:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )
        themes_match = re.search(
            r"Thematic Reinforcement:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )
        audience_match = re.search(
            r"Target Audience Appeal:?\s*(.*?)(?=\n\n|\n\d\.|\Z)",
            improvement_plan,
            re.DOTALL,
        )

        # Compile the improvement plan
        return {
            "full_plan": improvement_plan,
            "structure_improvements": (
                structure_improvements_match.group(1).strip()
                if structure_improvements_match
                else ""
            ),
            "plot_enhancement": plot_match.group(1).strip() if plot_match else "",
            "character_arc_refinements": (
                character_match.group(1).strip() if character_match else ""
            ),
            "pacing_adjustments": pacing_match.group(1).strip() if pacing_match else "",
            "key_scene_changes": scenes_match.group(1).strip() if scenes_match else "",
            "thematic_reinforcement": (
                themes_match.group(1).strip() if themes_match else ""
            ),
            "target_audience_appeal": (
                audience_match.group(1).strip() if audience_match else ""
            ),
        }

    def _apply_story_arc_refinements(
        self,
        content: str,
        improvement_plan: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply story arc refinements to the manuscript."""
        # Extract key scene changes from the improvement plan
        key_scene_changes = improvement_plan.get("key_scene_changes", "")

        # If no specific scene changes, return content unchanged
        if not key_scene_changes:
            return content

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Ensure the refinements will appeal to this audience.
            """

        # Create prompt for applying refinements
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Story Arc Refinement Specialist. Apply key improvements to enhance the story arcs in this manuscript.
        
        Key Scene Recommendations:
        {key_scene_changes}
        
        Plot Enhancement Recommendations:
        {plot_enhancement}
        
        {audience_context}
        
        Create 2-3 new or modified scenes (1-3 paragraphs each) that could be added to the manuscript to strengthen the story arcs.
        Each scene should:
        1. Address a specific issue identified in the recommendations
        2. Enhance character development, plot progression, or thematic depth
        3. Match the style and voice of the existing manuscript
        
        Format each scene as a separate, complete section that could be inserted into the manuscript.
        Indicate where in the narrative flow each scene would be placed (beginning, middle, or end).
        """
        )

        # Create the chain
        chain = (
            {
                "key_scene_changes": lambda _: key_scene_changes,
                "plot_enhancement": lambda _: improvement_plan.get(
                    "plot_enhancement", ""
                ),
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        new_scenes = chain.invoke("Generate new scenes")

        # Parse the scenes

        # Try to identify scene sections
        scene_blocks = re.split(r"(?:Scene\s+\d+:|New Scene\s+\d+:)", new_scenes)
        if scene_blocks and not scene_blocks[0].strip():
            scene_blocks = scene_blocks[1:]

        if not scene_blocks:
            return content

        # Process each scene
        updated_content = content

        for scene_block in scene_blocks:
            # Try to determine where to place this scene
            placement = "middle"  # Default
            if re.search(r"beginning|start|open|first", scene_block, re.IGNORECASE):
                placement = "beginning"
            elif re.search(r"end|final|conclusion|climax", scene_block, re.IGNORECASE):
                placement = "end"

            # Extract the scene content
            # Remove any placement instructions or headers
            scene_content = re.sub(
                r"^.*?(Placement|Location|Insert).*?\n",
                "",
                scene_block,
                flags=re.IGNORECASE,
            )
            scene_content = scene_content.strip()

            # Find an insertion point based on placement
            if placement == "beginning":
                # Insert after the first few paragraphs
                paragraphs = updated_content.split("\n\n")
                insert_idx = min(
                    5, len(paragraphs) // 10
                )  # After about 10% of paragraphs
                paragraphs.insert(insert_idx, scene_content)
                updated_content = "\n\n".join(paragraphs)

            elif placement == "end":
                # Insert before the last few paragraphs
                paragraphs = updated_content.split("\n\n")
                insert_idx = max(
                    len(paragraphs) - 5, int(len(paragraphs) * 0.9)
                )  # Before last 10% of paragraphs
                paragraphs.insert(insert_idx, scene_content)
                updated_content = "\n\n".join(paragraphs)

            else:  # middle
                # Insert near the middle
                mid_point = len(updated_content) // 2
                # Find the nearest paragraph break
                before_break = updated_content.rfind("\n\n", 0, mid_point)
                after_break = updated_content.find("\n\n", mid_point)

                if before_break != -1:
                    # Insert after this paragraph
                    updated_content = (
                        updated_content[:before_break]
                        + "\n\n"
                        + scene_content
                        + "\n\n"
                        + updated_content[before_break:]
                    )
                elif after_break != -1:
                    # Insert before this paragraph
                    updated_content = (
                        updated_content[:after_break]
                        + "\n\n"
                        + scene_content
                        + "\n\n"
                        + updated_content[after_break:]
                    )

        return updated_content
