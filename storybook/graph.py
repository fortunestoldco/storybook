from datetime import datetime 
import asyncio
from typing import Dict, Any, Literal, Union, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

from storybook.state import State, InputState, AgentOutput
from storybook.agents import (
    MarketResearcher, ContentAnalyzer, CharacterDeveloper,
    DialogueEnhancer, WorldBuilder, SubplotWeaver,
    StoryArcAnalyst, LanguagePolisher, QualityReviewer
)
from storybook.config import Configuration

async def research_team_supervisor(state: State, *, config: RunnableConfig, writer=None) -> Dict[str, Any]:
    """Coordinates and combines market and content analysis with streaming support."""
    agents = [MarketResearcher(config), ContentAnalyzer(config)]
    
    if writer:
        writer({"status": "Starting research team analysis"})
    
    results = await asyncio.gather(*[agent.process_manuscript(state.manuscript) for agent in agents])
    
    if writer:
        writer({"status": "Research team analysis complete"})
    
    return {
        "market_analysis": AgentOutput(content=results[0], timestamp=datetime.now(), agent_id="market_researcher"),
        "content_analysis": AgentOutput(content=results[1], timestamp=datetime.now(), agent_id="content_analyzer"),
        "current_step": "research_complete"
    }

async def creative_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates character, dialogue, world-building, and subplot teams."""
    agents = [
        CharacterDeveloper(config),
        DialogueEnhancer(config), 
        WorldBuilder(config),
        SubplotWeaver(config)
    ]
    
    results = await asyncio.gather(*[agent.process_manuscript(state.manuscript) for agent in agents])
    
    return {
        "characters": AgentOutput(content=results[0], timestamp=datetime.now(), agent_id="character_developer"),
        "dialogue": AgentOutput(content=results[1], timestamp=datetime.now(), agent_id="dialogue_enhancer"), 
        "world_building": AgentOutput(content=results[2], timestamp=datetime.now(), agent_id="world_builder"),
        "subplots": AgentOutput(content=results[3], timestamp=datetime.now(), agent_id="subplot_weaver"),
        "current_step": "creative_complete"
    }

async def quality_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates story arc, language, and quality review teams."""
    arc_agent = StoryArcAnalyst(config)
    lang_agent = LanguagePolisher(config)
    quality_agent = QualityReviewer(config)
    
    arc_result, lang_result = await asyncio.gather(
        arc_agent.process_manuscript(state.manuscript),
        lang_agent.process_manuscript(state.manuscript)
    )
    quality_result = await quality_agent.process_manuscript(state.manuscript)
    
    return {
        "story_arc": AgentOutput(content=arc_result, timestamp=datetime.now(), agent_id="story_arc_analyst"),
        "language": AgentOutput(content=lang_result, timestamp=datetime.now(), agent_id="language_polisher"),
        "quality_review": AgentOutput(content=quality_result, timestamp=datetime.now(), agent_id="quality_reviewer"),
        "current_step": "complete"
    }

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build and return the hierarchical storybook processing graph."""
    builder = StateGraph(State, input=InputState, config_schema=Configuration)
    
    # Add individual agent nodes
    agents = {
        "research_team": {
            "supervisor": "research_team_supervisor",
            "agents": ["market_researcher", "content_analyzer"]
        },
        "creative_team": {
            "supervisor": "creative_team_supervisor",
            "agents": ["character_developer", "dialogue_enhancer", "world_builder", "subplot_weaver"]
        },
        "quality_team": {
            "supervisor": "quality_team_supervisor",
            "agents": ["story_arc_analyst", "language_polisher", "quality_reviewer"]
        }
    }

    # Add all nodes and their connections
    for team, structure in agents.items():
        # Add supervisor
        builder.add_node(structure["supervisor"], globals()[structure["supervisor"]])
        
        # Add team agents and connect to supervisor
        for agent in structure["agents"]:
            builder.add_node(agent, globals()[agent])
            builder.add_edge(agent, structure["supervisor"])

    # Add high-level team routing
    builder.add_edge("__start__", "research_team_supervisor")
    builder.add_edge("research_team_supervisor", "creative_team_supervisor")
    builder.add_edge("creative_team_supervisor", "quality_team_supervisor")

    def should_revise(state: State) -> str:
        return "creative_team_supervisor" if state.quality_review.content.get("needs_revision", False) else "__end__"

    # Add conditional edges
    builder.add_conditional_edges(
        "quality_team_supervisor",
        path=should_revise
    )

    # Configure graph properties
    graph = builder.compile(
        checkpointer=MemorySaver(),
        debug=config.get("debug", False),
        interrupt_after=[f"{team}_supervisor" for team in agents.keys()]
    )
    
    graph.name = "DetailedStoryBookGraph"
    graph.stream_mode = "updates"
    return graph

__all__ = ["build_storybook"]