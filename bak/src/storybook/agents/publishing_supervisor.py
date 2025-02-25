"""Publishing Supervisor agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from storybook.config import AgentRole, SUPERVISOR_MODEL, USE_OLLAMA, OLLAMA_SUPERVISOR_MODEL
from storybook.tools import SUPERVISOR_TOOLS, PUBLISHING_TOOLS
from storybook.prompts import PUBLISHING_SUPERVISOR_SYSTEM_PROMPT, REVIEW_TASK_PROMPT
from storybook.utils import create_model_instance, extract_json_from_text, format_agent_response


def get_publishing_supervisor_agent(agent_id: str) -> AgentExecutor:
    """Create a publishing supervisor agent with appropriate tools."""
    llm = create_model_instance("supervisor", USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
        system_message=PUBLISHING_SUPERVISOR_SYSTEM_PROMPT,
    )

    agent = AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )

    return agent


def create_review_task(
    task_description: str,
    content_type: str,
    content_summary: str,
    review_criteria: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = "",
) -> str:
    """Create a formatted review task prompt."""
    return REVIEW_TASK_PROMPT.format(
        task_description=task_description,
        story_structure=story_structure,
        content_type=content_type,
        content_summary=content_summary,
        review_criteria=review_criteria,
        context=context,
        previous_feedback=previous_feedback or "No previous feedback available.",
    )


async def execute_publishing_review(
    agent_id: str,
    story_id: str,
    publishing_package: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = "",
) -> Dict[str, Any]:
    """Execute a publishing package review task with the supervisor agent."""
    try:
        # Create the review criteria
        review_criteria = (
            "1. Quality and appeal of metadata (title, description, keywords)\n"
            "2. Effectiveness of SEO strategy\n"
            "3. Appropriateness of categories and tags\n"
            "4. Engagement potential of promotional materials\n"
            "5. Overall presentation quality\n"
            "6. Alignment with target audience and platform requirements"
        )

        # Create the task prompt
        task_prompt = create_review_task(
            task_description="Review the publishing package for final approval.",
            content_type="Publishing Package",
            content_summary=publishing_package,
            review_criteria=review_criteria,
            story_structure=story_structure,
            context=context,
            previous_feedback=previous_feedback,
        )

        # Get the agent
        agent = get_publishing_supervisor_agent(agent_id)

        # Execute the task
        response = await agent.ainvoke(
            {"input": task_prompt, "agent_id": agent_id, "story_id": story_id}
        )

        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "publishing_review",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )
        else:
            return format_agent_response(
                content="Error: No output from supervisor agent", metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during publishing review: {str(e)}", metadata={"error": True}
        )
