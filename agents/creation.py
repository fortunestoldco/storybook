from typing import Dict, List
from langchain_core.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent
from langsmith.run_helpers import traceable
from tools.creation import (
    generate_content,
    review_content,
    manage_continuity
)

@traceable(name="Content Creator Agent")
def content_creator_agent(state: StoryState) -> Dict:
    """Agent responsible for generating story content."""
    tools = [
        Tool(name="generate_content", func=generate_content),
        Tool(name="manage_continuity", func=manage_continuity)
    ]
    
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=get_llm_from_state(state),
        tools=tools
    )
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    result = executor.invoke({
        "input": {
            "title": state["title"],
            "manuscript": state["manuscript"],
            "section": "current_chapter"
        }
    })
    
    return {
        "content": result.get("content", {}),
        "feedback": ["Content generated", "Continuity maintained"],
        "agent_type": "content_creator",
        "agent_model": state["model_name"]
    }

@traceable(name="Draft Reviewer Agent")
def draft_reviewer_agent(state: StoryState) -> Dict:
    """Agent responsible for reviewing generated content."""
    tools = [
        Tool(name="review_content", func=review_content),
        Tool(name="manage_continuity", func=manage_continuity)
    ]
    
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=get_llm_from_state(state),
        tools=tools
    )
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    result = executor.invoke({
        "input": {
            "title": state["title"],
            "manuscript": state["manuscript"]
        }
    })
    
    return {
        "review": result.get("review", {}),
        "feedback": ["Content reviewed", "Quality assessment completed"],
        "agent_type": "draft_reviewer",
        "agent_model": state["model_name"]
    }