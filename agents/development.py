from typing import Dict, List, Any
from langchain_core.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent
from langsmith.run_helpers import traceable
from tools.development import (
    develop_plot_structure,
    develop_characters,
    develop_world_building
)

@traceable(name="Plot Development Agent")
def plot_development_agent(state: StoryState) -> Dict:
    """Agent responsible for plot development and structure."""
    tools = [
        Tool(name="develop_plot_structure", func=develop_plot_structure),
        Tool(name="develop_world_building", func=develop_world_building)
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
            "phase": "plot_development"
        }
    })
    
    return {
        "plot_structure": result.get("plot_elements", {}),
        "feedback": ["Plot structure developed", "World-building integrated"],
        "agent_type": "plot_developer",
        "agent_model": state["model_name"]
    }

@traceable(name="Character Development Agent")
def character_development_agent(state: StoryState) -> Dict:
    """Agent responsible for character development."""
    tools = [
        Tool(name="develop_characters", func=develop_characters),
        Tool(name="develop_world_building", func=develop_world_building)
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
            "phase": "character_development"
        }
    })
    
    return {
        "character_profiles": result.get("character_development", {}),
        "feedback": ["Character profiles created", "Character arcs defined"],
        "agent_type": "character_developer",
        "agent_model": state["model_name"]
    }