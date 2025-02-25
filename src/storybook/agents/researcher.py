"""Researcher agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import AgentRole, DEFAULT_MODEL, RESEARCH_MODEL, USE_OLLAMA, OLLAMA_RESEARCH_MODEL
from agents.tools import RESEARCH_TOOLS
from agents.prompts import RESEARCHER_SYSTEM_PROMPT, RESEARCH_TASK_PROMPT
from agents.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_researcher_agent(agent_id: str) -> AgentExecutor:
    """Create a researcher agent with appropriate tools."""
    llm = create_model_instance("research", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=RESEARCH_TOOLS,
        system_message=RESEARCHER_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=RESEARCH_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent

def create_research_task(
    task_description: str,
    story_request: str,
    story_structure: str,
    research_focus: str,
    bible_entries: str = "",
    existing_research: str = ""
) -> str:
    """Create a formatted research task prompt."""
    return RESEARCH_TASK_PROMPT.format(
        task_description=task_description,
        story_request=story_request,
        story_structure=story_structure,
        research_focus=research_focus,
        bible_entries=bible_entries or "No bible entries available yet.",
        existing_research=existing_research or "No prior research exists for this story."
    )

async def execute_research_task(
    agent_id: str,
    task_description: str,
    story_id: str,
    story_request: str,
    story_structure: str,
    research_focus: str,
    bible_entries: str = "",
    existing_research: str = ""
) -> Dict[str, Any]:
    """Execute a research task with the researcher agent."""
    try:
        # Create the task prompt
        task_prompt = create_research_task(
            task_description=task_description,
            story_request=story_request,
            story_structure=story_structure,
            research_focus=research_focus,
            bible_entries=bible_entries,
            existing_research=existing_research
        )
        
        # Get the agent
        agent = get_researcher_agent(agent_id)
        
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
                    "task_type": "research",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from research agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during research: {str(e)}",
            metadata={"error": True}
        )
