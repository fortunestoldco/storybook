from typing import Dict, List
from langchain.agents import AgentExecutor, StructuredChatAgent
from langsmith.run_helpers import traceable
from tools.analysis import (
    analyze_story_structure,
    evaluate_market_fit,
    assess_character_development
)
from tools.project_management import create_project_timeline, analyze_market_trends

@traceable(name="Quality Assessment Director")
def quality_assessment_director_agent(state: StoryState) -> Dict:
    """Quality Assessment Director agent for evaluating story quality."""
    tools = [
        Tool(name="analyze_story_structure", func=analyze_story_structure),
        Tool(name="evaluate_market_fit", func=evaluate_market_fit),
        Tool(name="assess_character_development", func=assess_character_development)
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
            "criteria": ["structure", "market", "characters"]
        }
    })
    
    return {
        "feedback": result["output"],
        "scores": {
            "structure": result.get("structure_score", 0.0),
            "market": result.get("market_score", 0.0),
            "characters": result.get("character_score", 0.0)
        },
        "agent_type": "quality_assessment_director",
        "agent_model": state["model_name"]
    }

@traceable(name="Project Timeline Manager")
def project_timeline_manager_agent(state: StoryState) -> Dict:
    """Project Timeline Manager agent for scheduling and coordination."""
    tools = [
        Tool(name="create_project_timeline", func=create_project_timeline),
        Tool(name="analyze_market_trends", func=analyze_market_trends)
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
            "target_completion": None,  # Could be added to state if needed
            "milestones": ["Development", "Creation", "Refinement"]
        }
    })
    
    return {
        "timeline": result.get("timeline", {}),
        "feedback": ["Timeline created and project milestones set"],
        "agent_type": "project_timeline_manager",
        "agent_model": state["model_name"]
    }

@traceable(name="Market Alignment Director")
def market_alignment_director_agent(state: StoryState) -> Dict:
    """Market Alignment Director agent for market positioning."""
    tools = [
        Tool(name="analyze_market_trends", func=analyze_market_trends),
        Tool(name="evaluate_market_fit", func=evaluate_market_fit)
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
        "market_analysis": result.get("market_analysis", {}),
        "feedback": ["Market alignment analysis completed"],
        "agent_type": "market_alignment_director",
        "agent_model": state["model_name"]
    }