"""Writer agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from storybook.config import (
    AgentRole,
    WRITING_MODEL,
    USE_OLLAMA,
    OLLAMA_WRITING_MODEL,
    StoryStructure,
)
from storybook.tools import WRITING_TOOLS
from storybook.prompts import (
    WRITER_SYSTEM_PROMPT,
    WRITING_TASK_PROMPT,
    THREE_ACT_STRUCTURE_PROMPT,
    FIVE_ACT_STRUCTURE_PROMPT,
    HEROS_JOURNEY_STRUCTURE_PROMPT,
)
from storybook.utils import create_model_instance, extract_json_from_text, format_agent_response


def get_structure_prompt(structure: str) -> str:
    """Get the appropriate structure prompt based on the story structure."""
    if structure == StoryStructure.THREE_ACT.value:
        return THREE_ACT_STRUCTURE_PROMPT
    elif structure == StoryStructure.FIVE_ACT.value:
        return FIVE_ACT_STRUCTURE_PROMPT
    elif structure == StoryStructure.HEROS_JOURNEY.value:
        return HEROS_JOURNEY_STRUCTURE_PROMPT
    else:
        return ""


def get_writer_agent(agent_id: str, story_structure: str = None) -> AgentExecutor:
    """Create a writer agent with appropriate tools."""
    llm = create_model_instance("writing", USE_OLLAMA)

    # Customize system prompt based on story structure
    system_prompt = WRITER_SYSTEM_PROMPT
    structure_prompt = get_structure_prompt(story_structure) if story_structure else ""

    if structure_prompt:
        system_prompt = f"{system_prompt}\n\n{structure_prompt}"

    prompt = create_openai_tools_agent(llm=llm, tools=WRITING_TOOLS, system_message=system_prompt)

    agent = AgentExecutor(
        agent=prompt,
        tools=WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=12,  # Writers may need more iterations
    )

    return agent


def create_writing_task(
    task_description: str,
    story_request: str,
    section: str,
    story_structure: str = "",
    target_length: str = "",
    tone_style: str = "",
    reference_materials: str = "",
    bible_entries: str = "",
    outline_elements: str = "",
    previous_feedback: str = "",
) -> str:
    """Create a formatted writing task prompt."""
    return WRITING_TASK_PROMPT.format(
        task_description=task_description,
        story_request=story_request,
        story_structure=story_structure,
        section=section,
        target_length=target_length or "Appropriate length for this section",
        tone_style=tone_style or "Not specified",
        reference_materials=reference_materials or "No specific reference materials provided.",
        bible_entries=bible_entries or "No bible entries available yet.",
        outline_elements=outline_elements or "No outline elements specified.",
        previous_feedback=previous_feedback or "No previous feedback available.",
    )


async def execute_writing_task(
    agent_id: str,
    story_id: str,
    section_id: str,
    task_description: str,
    story_request: str,
    story_structure: str = "",
    target_length: str = "",
    tone_style: str = "",
    reference_materials: str = "",
    bible_entries: str = "",
    outline_elements: str = "",
    previous_feedback: str = "",
) -> Dict[str, Any]:
    """Execute a writing task with the writer agent."""
    try:
        # Create the task prompt
        task_prompt = create_writing_task(
            task_description=task_description,
            story_request=story_request,
            section=section_id,
            story_structure=story_structure,
            target_length=target_length,
            tone_style=tone_style,
            reference_materials=reference_materials,
            bible_entries=bible_entries,
            outline_elements=outline_elements,
            previous_feedback=previous_feedback,
        )

        # Get the agent
        agent = get_writer_agent(agent_id, story_structure)

        # Execute the task
        response = await agent.ainvoke(
            {
                "input": task_prompt,
                "agent_id": agent_id,
                "story_id": story_id,
                "section_id": section_id,
            }
        )

        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "writing",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "section_id": section_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )
        else:
            return format_agent_response(
                content="Error: No output from writer agent", metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during writing: {str(e)}", metadata={"error": True}
        )
