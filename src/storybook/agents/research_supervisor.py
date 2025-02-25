"""Research Supervisor agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import AgentRole, SUPERVISOR_MODEL, USE_OLLAMA, OLLAMA_SUPERVISOR_MODEL
from agents.tools import SUPERVISOR_TOOLS, RESEARCH_TOOLS
from agents.prompts import RESEARCH_SUPERVISOR_SYSTEM_PROMPT, REVIEW_TASK_PROMPT
from agents.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_research_supervisor_agent(agent_id: str) -> AgentExecutor:
    """Create a research supervisor agent with appropriate tools."""
    llm = create_model_instance("supervisor", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        system_message=RESEARCH_SUPERVISOR_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
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

async def execute_research_review(
    agent_id: str,
    story_id: str,
    research_content: str,
    story_structure: str = "",
    context: str = "",
    previous_feedback: str = ""
) -> Dict[str, Any]:
    """Execute a research review task with the supervisor agent."""
    try:
        # Create the review criteria
        review_criteria = (
            "1. Comprehensiveness - Does the research cover all necessary areas?\n"
            "2. Relevance - Is the research directly applicable to the story requirements?\n"
            "3. Accuracy - Does the research appear accurate and from reliable sources?\n"
            "4. Usability - Is the research organized in a way that writers can easily use?\n"
            "5. Structure Support - Does the research adequately support the story structure?\n"
            "6. Gaps - Are there any obvious gaps or missing areas of research?"
        )
        
        # Create the task prompt
        task_prompt = create_review_task(
            task_description=f"Review the research conducted for the story with the {story_structure} structure.",
            content_type="Research",
            content_summary=research_content,
            review_criteria=review_criteria,
            story_structure=story_structure,
            context=context,
            previous_feedback=previous_feedback
        )
        
        # Get the agent
        agent = get_research_supervisor_agent(agent_id)
        
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
                    "task_type": "research_review",
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
            content=f"Error during research review: {str(e)}",
            metadata={"error": True}
        )
