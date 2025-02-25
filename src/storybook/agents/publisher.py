"""Publisher agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import AgentRole, PUBLISHING_MODEL, USE_OLLAMA, OLLAMA_PUBLISHING_MODEL
from agents.tools import PUBLISHING_TOOLS
from agents.prompts import PUBLISHER_SYSTEM_PROMPT, PUBLISHING_TASK_PROMPT
from agents.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_publisher_agent(agent_id: str) -> AgentExecutor:
    """Create a publisher agent with appropriate tools."""
    llm = create_model_instance("publishing", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=PUBLISHING_TOOLS,
        system_message=PUBLISHER_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=PUBLISHING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8
    )
    
    return agent

def create_publishing_task(
    task_description: str,
    title: str,
    genre: str,
    target_audience: str,
    length: str,
    story_structure: str = "",
    content_summary: str = "",
    publishing_platforms: str = "",
    seo_goals: str = ""
) -> str:
    """Create a formatted publishing task prompt."""
    return PUBLISHING_TASK_PROMPT.format(
        task_description=task_description,
        title=title,
        genre=genre,
        target_audience=target_audience,
        length=length,
        story_structure=story_structure,
        content_summary=content_summary or "Content summary not provided.",
        publishing_platforms=publishing_platforms or "Website, blog, or other digital platforms as appropriate.",
        seo_goals=seo_goals or "Maximize discoverability through effective keywords, meta description, and categorization."
    )

async def execute_publishing_task(
    agent_id: str,
    story_id: str,
    title: str,
    genre: str,
    target_audience: str,
    content: str,
    task_description: str = "Prepare the story for publication by creating metadata, formatting content, and developing promotional materials.",
    story_structure: str = "",
    publishing_platforms: str = "",
    seo_goals: str = ""
) -> Dict[str, Any]:
    """Execute a publishing task with the publisher agent."""
    try:
        # Calculate approximate length
        length = f"Approximately {len(content)} characters ({len(content.split())} words)"
        
        # Create content summary (truncated for prompt)
        content_summary = content[:2000] + "..." if len(content) > 2000 else content
        
        # Create the task prompt
        task_prompt = create_publishing_task(
            task_description=task_description,
            title=title,
            genre=genre,
            target_audience=target_audience,
            length=length,
            story_structure=story_structure,
            content_summary=content_summary,
            publishing_platforms=publishing_platforms,
            seo_goals=seo_goals
        )
        
        # Get the agent
        agent = get_publisher_agent(agent_id)
        
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
                    "task_type": "publishing",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from publisher agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during publishing preparation: {str(e)}",
            metadata={"error": True}
        )

async def execute_publish_action(
    agent_id: str,
    story_id: str,
    title: str,
    content: str,
    metadata: Dict[str, Any],
    platform: str = "website"
) -> Dict[str, Any]:
    """Execute the final publishing action."""
    try:
        # Create the publishing prompt
        prompt = f"""
        # Story Publishing Action
        
        The story and publishing package have been approved for publication. Please execute the publishing process.
        
        ## Story Details
        Title: {title}
        ID: {story_id}
        Length: {len(content.split())} words
        
        ## Publishing Platform
        {platform}
        
        ## Metadata
        {json.dumps(metadata, indent=2)}
        
        1. Confirm all metadata and formatting are complete
        2. Publish the story to {platform}
        3. Return the publishing details including URLs and timestamps
        """
        
        # Get the agent
        agent = get_publisher_agent(agent_id)
        
        # Execute the publishing action
        response = await agent.ainvoke({
            "input": prompt,
            "agent_id": agent_id,
            "story_id": story_id,
            "platform": platform
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "execute_publishing",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "platform": platform,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from publisher agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during publishing execution: {str(e)}",
            metadata={"error": True}
        )
