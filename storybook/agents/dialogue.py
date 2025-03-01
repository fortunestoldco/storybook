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

            # Extract dialogue sections
            dialogue_sections = self._extract_dialogue_sections(manuscript["content"])

            # Enhance each dialogue section
            enhanced_dialogue = []
            for section in dialogue_sections:
                enhanced_section = self._enhance_section(
                    section, target_audience, research_insights
                )
                enhanced_dialogue.append(enhanced_section)

            # Combine enhanced sections back into the manuscript
            enhanced_content = self._combine_enhanced_sections(
                manuscript["content"], enhanced_dialogue
            )

            return {
                "manuscript_id": manuscript_id,
                "message": "Dialogue enhancement complete",
                "enhanced_content": enhanced_content
            }

        except Exception as e:
            logger.error(f"Error in enhance_dialogue: {str(e)}")
            return self.handle_error(e)

    def process_manuscript(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]], research_insights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process manuscript for dialogue enhancement."""
        try:
            return self.enhance_dialogue(
                manuscript_id,
                target_audience,
                research_insights
            )
        except Exception as e:
            logger.error(f"Error in dialogue enhancement: {str(e)}")
            return self.handle_error(e)

    def _extract_dialogue_sections(self, content: str) -> List[str]:
        """Extract dialogue sections from the manuscript."""
        dialogue_sections = re.findall(r'\"(.*?)\"', content, re.DOTALL)
        return dialogue_sections

    def _enhance_section(
        self,
        section: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance a single dialogue section."""
        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Reading Preferences: {target_audience.get('reading_preferences', {}).get('dialogue', 'Various preferences')}
            
            Consider the target audience preferences when enhancing the dialogue.
            """

        # Define the prompt
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Dialogue Enhancement Specialist. Enhance the following dialogue section to make it more engaging and natural.
        
        Original Dialogue:
        {dialogue_section}
        
        {audience_context}
        
        Enhance the dialogue while maintaining the original meaning and context. Focus on improving the flow, adding character-specific nuances, and making it more engaging for the target audience.
        """
        )

        # Create the chain
        chain = (
            {
                "dialogue_section": lambda _: section,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        enhanced_section = chain.invoke(section)

        return enhanced_section

    def _combine_enhanced_sections(self, original_content: str, enhanced_sections: List[str]) -> str:
        """Combine enhanced dialogue sections back into the manuscript."""
        enhanced_content = original_content
        for original, enhanced in zip(self._extract_dialogue_sections(original_content), enhanced_sections):
            enhanced_content = enhanced_content.replace(original, enhanced, 1)
        return enhanced_content
