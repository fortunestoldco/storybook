"""Human-in-Loop agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from storybook.config import AgentRole
from storybook.prompts import HUMAN_REVIEW_PROMPT
from storybook.utils import format_agent_response, prepare_human_review_prompt

def create_human_review_prompt(
    review_type: str,
    item_description: str,
    options: List[Dict[str, Any]],
    story_structure: str = "",
    context: str = "",
    considerations: str = "",
    decision_requested: str = "",
    reason_for_human_review: str = ""
) -> str:
    """Create a formatted human review prompt."""
    return HUMAN_REVIEW_PROMPT.format(
        review_type=review_type,
        story_structure=story_structure,
        item_description=item_description,
        context=context or "No additional context provided.",
        options="\n".join([f"{i+1}. {option['text']}: {option.get('description', '')}" 
                          for i, option in enumerate(options)]),
        considerations=considerations or "Consider the overall
