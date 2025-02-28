from __future__ import annotations

# Standard library imports
from typing import Dict, Any, Optional, List
import logging
import json
import re
from datetime import datetime

# Third-party imports
from langchain_core.language_models import BaseChatModel

# Local imports
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents with configurable LLM support."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        self.llm = create_llm(llm_config) if llm_config else get_llm()
        self.document_store = DocumentStore()

    def update_llm(self, llm_config: Dict[str, Any]) -> None:
        """Update LLM configuration at runtime."""
        self.llm = create_llm(llm_config)

    def get_tools(self) -> list:
        """Get tools available to this agent. Override in subclasses."""
        return []

    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        manuscript_id = kwargs.get('manuscript_id')
        if not manuscript_id:
            logger.error("Missing required manuscript_id")
            return False
        return True

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors in agent execution."""
        logger.error(f"Agent error: {str(error)}")
        return {
            "status": "error",
            "message": str(error),
            "type": error.__class__.__name__
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs using the configured LLM."""
        try:
            # Update LLM config if provided in inputs
            if "llm_config" in inputs:
                self.update_llm(inputs["llm_config"])
            return self.process_manuscript(
                inputs["manuscript_id"],
                inputs.get("target_audience"),
                inputs.get("research_insights")
            )
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} process: {str(e)}")
            return self.handle_error(e)

    def process_manuscript(self, manuscript_id: str, target_audience: Optional[Dict[str, Any]], research_insights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process manuscript for [agent specific task]."""
        try:
            if not self.validate_input(manuscript_id=manuscript_id):
                return {"error": "Invalid manuscript_id"}
                
            manuscript = self.document_store.get_manuscript(manuscript_id)
            if not manuscript:
                return {"error": f"Manuscript {manuscript_id} not found"}

            # Agent-specific processing...
            result = self._agent_specific_processing(manuscript["content"])

            return {
                "manuscript_id": manuscript_id,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return self.handle_error(e)