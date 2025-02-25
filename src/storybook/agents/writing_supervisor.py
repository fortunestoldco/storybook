"""Writing Supervisor agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from storybook.config import AgentRole, SUPERVISOR_MODEL, USE_OLLAMA, OLLAMA_SUPERVISOR_MODEL
from storybook.tools import SUPERVISOR_TOOLS, WRITING_TOOLS
from storybook.prompts import WRITING_SUPERVISOR_SYSTEM_PROMPT, REVIEW_TASK_PROMPT
from storybook.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_writing_supervisor_agent(agent_id: str) -> AgentExecutor:
    """Create a writing supervisor agent with appropriate tools."""
    llm = create_model_instance("supervisor", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        system_message=WRITING_SUPERVISOR_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8
    )
    
    return agent

def create_review_task(
    task_description: str,
    content_type: str,
    content_summary: str,
    review_criteria: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = ""
) -> str:
    """Create a formatted review task prompt."""
    return REVIEW_TASK_PROMPT.format(
        task_description=task_description,
        story_structure=story_structure,
        content_type=content_type,
        content_summary=content_summary,
        review_criteria=review_criteria,
        context=context,
        previous_feedback=previous_feedback or "No previous feedback available."
    )

async def execute_outline_review(
    agent_id: str,
    story_id: str,
    outline_content: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = ""
) -> Dict[str, Any]:
    """Execute an outline review task with the supervisor agent."""
    try:
        # Create the review criteria with structure-specific elements
        structure_reference = f"adherence to the {story_structure} structure" if story_structure else "structural coherence"
        
        review_criteria = (
            f"1. {structure_reference.capitalize()}\n"
            "2. Adherence to client requirements\n"
            "3. Character development and dimension\n"
            "4. Plot coherence and engagement\n"
            "5. Setting detail and integration\n"
            "6. Theme depth and resonance\n"
            "7. Overall creative quality and potential"
        )
        
        # Create the task prompt
        task_prompt = create_review_task(
            task_description=f"Review the story outline for quality, {structure_reference}, and alignment with requirements.",
            content_type="Story Outline",
            content_summary=outline_content,
            review_criteria=review_criteria,
            story_structure=story_structure,
            context=context,
            previous_feedback=previous_feedback
        )
        
        # Get the agent
        agent = get_writing_supervisor_agent(agent_id)
        
        # Execute the task
        response = await agent.ainvoke({
            "input": task_prompt,
            "agent_id": agent_id,
            "story_id": story_id
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "outline_review",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from supervisor agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during outline review: {str(e)}",
            metadata={"error": True}
        )

async def execute_draft_review(
    agent_id: str,
    story_id: str,
    draft_content: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = ""
) -> Dict[str, Any]:
    """Execute a draft review task with the supervisor agent."""
    try:
        # Create the review criteria
        review_criteria = (
            "1. Overall quality and readability\n"
            "2. Adherence to client requirements and outline\n"
            f"3. Alignment with {story_structure} structure\n"
            "4. Character consistency and development\n"
            "5. Plot coherence and engagement\n"
            "6. Pacing and flow\n"
            "7. Editorial quality (grammar, spelling, punctuation)\n"
            "8. Thematic resonance"
        )
        
        # Create the task prompt
        task_prompt = create_review_task(
            task_description="Review the edited story draft for final approval.",
            content_type="Edited Story Draft",
            content_summary=draft_content[:5000] + "..." if len(draft_content) > 5000 else draft_content,
            review_criteria=review_criteria,
            story_structure=story_structure,
            context=context,
            previous_feedback=previous_feedback
        )
        
        # Get the agent
        agent = get_writing_supervisor_agent(agent_id)
        
        # Execute the task
        response = await agent.ainvoke({
            "input": task_prompt,
            "agent_id": agent_id,
            "story_id": story_id
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "draft_review",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from supervisor agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during draft review: {str(e)}",
            metadata={"error": True}
        )
