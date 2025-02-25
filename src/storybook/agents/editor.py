"""Editor agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import AgentRole, WRITING_MODEL, USE_OLLAMA, OLLAMA_WRITING_MODEL
from agents.tools import EDITING_TOOLS
from agents.prompts import EDITOR_SYSTEM_PROMPT, EDITING_TASK_PROMPT
from agents.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_editor_agent(agent_id: str) -> AgentExecutor:
    """Create an editor agent with appropriate tools."""
    llm = create_model_instance("writing", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=EDITING_TOOLS,
        system_message=EDITOR_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=EDITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent

def create_editing_task(
    task_description: str,
    content: str,
    story_structure: str = "",
    bible_reference: str = "",
    focus_areas: str = "",
    previous_feedback: str = "",
    style_guide: str = ""
) -> str:
    """Create a formatted editing task prompt."""
    return EDITING_TASK_PROMPT.format(
        task_description=task_description,
        story_structure=story_structure,
        content=content,
        bible_reference=bible_reference or "No bible references available.",
        focus_areas=focus_areas or "General editing for quality and consistency.",
        previous_feedback=previous_feedback or "No previous feedback available.",
        style_guide=style_guide or "Follow standard editing practices."
    )

async def execute_editing_task(
    agent_id: str,
    story_id: str,
    content: str,
    task_description: str = "Edit the story draft for grammar, style, consistency, and overall quality.",
    story_structure: str = "",
    bible_reference: str = "",
    focus_areas: str = "",
    previous_feedback: str = "",
    style_guide: str = ""
) -> Dict[str, Any]:
    """Execute an editing task with the editor agent."""
    try:
        # Create default focus areas if not provided
        if not focus_areas:
            focus_areas = (
                "1. Grammar, spelling, and punctuation\n"
                "2. Sentence structure and flow\n"
                "3. Consistency in tone, character voices, and details\n"
                "4. Plot coherence and pacing\n"
                "5. Adherence to story outline and bible\n"
                "6. Overall readability and engagement"
            )
        
        # Create the task prompt
        task_prompt = create_editing_task(
            task_description=task_description,
            content=content,
            story_structure=story_structure,
            bible_reference=bible_reference,
            focus_areas=focus_areas,
            previous_feedback=previous_feedback,
            style_guide=style_guide
        )
        
        # Get the agent
        agent = get_editor_agent(agent_id)
        
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
                    "task_type": "editing",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from editor agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during editing: {str(e)}",
            metadata={"error": True}
        )
