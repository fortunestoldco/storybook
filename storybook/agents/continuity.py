from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
import json
import re
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from storybook.agents.base import BaseAgent  # Fix typo in BaseAgentnt
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore  # Change from mongodb_store
from storybook.tools.document_tools import DocumentTools

logger = logging.getLogger(__name__)

class ContinuityEditor(BaseAgent):
    """Agent responsible for maintaining narrative continuity."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.document_tools = DocumentTools()

    def process_manuscript(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]], research_insights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process manuscript for continuity checking."""
        try:
            return self.check_continuity(
                manuscript_id,
                target_audience,
                research_insights
            )
        except Exception as e:
            logger.error(f"Error in continuity checking: {str(e)}")
            return self.handle_error(e)

    def check_continuity(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]] = None, research_insights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check narrative continuity in the manuscript."""
        try:
            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            content = manuscript.get("content", "")

            # Define the prompt for continuity checking
            prompt = ChatPromptTemplate.from_template(
                """
                You are a Continuity Editor. Analyze the following manuscript for narrative continuity issues.
                Identify any inconsistencies in plot, character development, setting, and timeline.
                Provide a detailed report highlighting the issues and suggesting possible resolutions.

                Manuscript Content:
                {content}
                """
            )

            # Create the chain
            chain = (
                {"content": lambda _: content}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            continuity_report = chain.invoke("Check continuity")

            return {
                "manuscript_id": manuscript_id,
                "continuity_report": continuity_report,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in check_continuity: {str(e)}")
            return self.handle_error(e)
