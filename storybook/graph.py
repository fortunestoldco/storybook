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
    if writer:
        writer({"status": "Starting research team analysis"})
    
    market_result = await market_researcher_node(state, config=config)
    content_result = await content_analyzer_node(state, config=config)
    
    if writer:
        writer({"status": "Research team analysis complete"})
    
    return {
        "market_analysis": AgentOutput(content=market_result["market_analysis"], timestamp=datetime.now(), agent_id="market_researcher"),
        "content_analysis": AgentOutput(content=content_result["content_analysis"], timestamp=datetime.now(), agent_id="content_analyzer"),
        "current_step": "research_complete"
    }

async def creative_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates character, dialogue, world-building, and subplot teams."""
    chars = await character_developer_node(state, config=config)
    dialog = await dialogue_enhancer_node(state, config=config)
    world = await world_builder_node(state, config=config)
    subplot = await subplot_weaver_node(state, config=config)
    
    return {
        "characters": AgentOutput(content=chars["characters"], timestamp=datetime.now(), agent_id="character_developer"),
        "dialogue": AgentOutput(content=dialog["dialogue"], timestamp=datetime.now(), agent_id="dialogue_enhancer"),
        "world_building": AgentOutput(content=world["world_building"], timestamp=datetime.now(), agent_id="world_builder"),
        "subplots": AgentOutput(content=subplot["subplots"], timestamp=datetime.now(), agent_id="subplot_weaver"),
        "current_step": "creative_complete"
    }

async def quality_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates story arc, language, and quality review teams."""
    arc_result = await story_arc_analyst_node(state, config=config)
    lang_result = await language_polisher_node(state, config=config)
    quality_result = await quality_reviewer_node(state, config=config)
    
    return {
        "story_arc": AgentOutput(content=arc_result["story_arc"], timestamp=datetime.now(), agent_id="story_arc_analyst"),
        "language": AgentOutput(content=lang_result["language"], timestamp=datetime.now(), agent_id="language_polisher"),
        "quality_review": AgentOutput(content=quality_result["quality_review"], timestamp=datetime.now(), agent_id="quality_reviewer"),
        "current_step": "complete"
    }

# Add individual agent node functions
async def market_researcher_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Market researcher agent node."""
    agent = MarketResearcher(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"market_analysis": result}

async def content_analyzer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Content analyzer agent node."""
    agent = ContentAnalyzer(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"content_analysis": result}

async def character_developer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = CharacterDeveloper(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"characters": result}

async def dialogue_enhancer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = DialogueEnhancer(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"dialogue": result}

async def world_builder_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = WorldBuilder(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"world_building": result}

async def subplot_weaver_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = SubplotWeaver(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"subplots": result}

async def story_arc_analyst_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = StoryArcAnalyst(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"story_arc": result}

async def language_polisher_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = LanguagePolisher(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"language": result}

async def quality_reviewer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    agent = QualityReviewer(config)
    result = await agent.process_manuscript(state.manuscript)
    return {"quality_review": result}

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build and return the hierarchical storybook processing graph."""
    builder = StateGraph(State, input=InputState, config_schema=Configuration)
    
    # Add nodes
    team_structure = {
        "research_team_supervisor": ["market_researcher", "content_analyzer"],
        "creative_team_supervisor": ["character_developer", "dialogue_enhancer", "world_builder", "subplot_weaver"],
        "quality_team_supervisor": ["story_arc_analyst", "language_polisher", "quality_reviewer"]
    }

    # Add all nodes first
    for supervisor, agents in team_structure.items():
        builder.add_node(supervisor, globals()[supervisor])
        for agent in agents:
            builder.add_node(agent, globals()[f"{agent}_node"])

    # Add team workflow edges
    builder.add_edge("__start__", "research_team_supervisor")
    builder.add_edge("research_team_supervisor", "creative_team_supervisor")
    builder.add_edge("creative_team_supervisor", "quality_team_supervisor")

    def should_revise(state: State) -> str:
        return "creative_team_supervisor" if state.quality_review.content.get("needs_revision", False) else "__end__"

    builder.add_conditional_edges(
        "quality_team_supervisor",
        path=should_revise
    )

    graph = builder.compile(
        checkpointer=MemorySaver(),
        debug=config.get("debug", False),
        interrupt_after=[f"{team}_supervisor" for team in team_structure.keys()]
    )
    
    graph.name = "DetailedStoryBookGraph"
    graph.stream_mode = "updates"
    return graph

__all__ = ["build_storybook"]