"""Joint Writer agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from storybook.config import AgentRole, WRITING_MODEL, StoryStructure
from storybook.tools import WRITING_TOOLS
from storybook.prompts import (
    JOINT_WRITER_SYSTEM_PROMPT, JOINT_WRITING_TASK_PROMPT, 
    THREE_ACT_STRUCTURE_PROMPT, FIVE_ACT_STRUCTURE_PROMPT, HEROS_JOURNEY_STRUCTURE_PROMPT
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

def get_joint_writer_agent(agent_id: str, story_structure: str = None) -> AgentExecutor:
    """Create a joint writer agent with appropriate tools."""
    # Joint writers always use the API model for best quality, regardless of USE_OLLAMA setting
    llm = create_model_instance("writing", False)
    
    # Customize system prompt based on story structure
    system_prompt = JOINT_WRITER_SYSTEM_PROMPT
    structure_prompt = get_structure_prompt(story_structure) if story_structure else ""
    
    if structure_prompt:
        system_prompt = f"{system_prompt}\n\n{structure_prompt}"
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=WRITING_TOOLS,
        system_message=system_prompt
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15  # Joint writers get more iterations for complex sections
    )
    
    return agent

def create_joint_writing_task(
    task_description: str,
    story_request: str,
    section: str,
    complexity_factors: str,
    story_structure: str = "",
    target_length: str = "",
    tone_style: str = "",
    reference_materials: str = "",
    bible_entries: str = "",
    outline_elements: str = "",
    previous_contributions: str = ""
) -> str:
    """Create a formatted joint writing task prompt."""
    return JOINT_WRITING_TASK_PROMPT.format(
        task_description=task_description,
        story_request=story_request,
        story_structure=story_structure,
        section=section,
        complexity_factors=complexity_factors,
        target_length=target_length or "Appropriate length for this section",
        tone_style=tone_style or "Not specified",
        reference_materials=reference_materials or "No specific reference materials provided.",
        bible_entries=bible_entries or "No bible entries available yet.",
        outline_elements=outline_elements or "No outline elements specified.",
        previous_contributions=previous_contributions or "No previous contributions available."
    )

async def execute_joint_writing_task(
    agent_id: str,
    story_id: str,
    section_id: str,
    component_writers: List[str],
    task_description: str,
    story_request: str,
    complexity_factors: str,
    story_structure: str = "",
    target_length: str = "",
    tone_style: str = "",
    reference_materials: str = "",
    bible_entries: str = "",
    outline_elements: str = "",
    previous_contributions: Dict[str, str] = None
) -> Dict[str, Any]:
    """Execute a joint writing task with the joint writer agent."""
    try:
        # Format previous contributions
        formatted_contributions = "No previous contributions."
        
        if previous_contributions:
            formatted_contributions = "## Previous Contributions\n\n"
            for writer_id, content in previous_contributions.items():
                formatted_contributions += f"### From Writer {writer_id}\n{content}\n\n"
        
        # Create the task prompt
        task_prompt = create_joint_writing_task(
            task_description=task_description,
            story_request=story_request,
            section=section_id,
            complexity_factors=complexity_factors,
            story_structure=story_structure,
            target_length=target_length,
            tone_style=tone_style,
            reference_materials=reference_materials,
            bible_entries=bible_entries,
            outline_elements=outline_elements,
            previous_contributions=formatted_contributions
        )
        
        # Get the agent
        agent = get_joint_writer_agent(agent_id, story_structure)
        
        # Execute the task
        response = await agent.ainvoke({
            "input": task_prompt,
            "agent_id": agent_id,
            "story_id": story_id,
            "section_id": section_id,
            "component_writers": component_writers
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "joint_writing",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "section_id": section_id,
                    "component_writers": component_writers,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from joint writer agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during joint writing: {str(e)}",
            metadata={"error": True}
        )
