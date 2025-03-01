from datetime import datetime 
import asyncio
from typing import Dict, Any, Literal, Union, Optional
from langchain import graphs
from langchain.graphs import Graph as LangChainGraph
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from storybook.config import Configuration, get_default_config

from storybook.state import State, InputState, AgentOutput
from storybook.agents import (
    MarketResearcher, ContentAnalyzer, CharacterDeveloper,
    DialogueEnhancer, WorldBuilder, SubplotWeaver,
    StoryArcAnalyst, ProseSpecialist, QualityReviewer
)

async def research_team_supervisor(state: State, *, config: RunnableConfig, writer=None) -> Dict[str, Any]:
    """Coordinates and combines market and content analysis with streaming support."""
    try:
        if writer:
            writer({"status": "Starting research team analysis"})
        
        market_result = await marketresearch_node(state, config=config)
        if "error" in market_result:
            raise ValueError(market_result["error"])
            
        content_result = await content_analyzer_node(state, config=config)
        if "error" in content_result:
            raise ValueError(content_result["error"])
        
        if writer:
            writer({"status": "Research team analysis complete"})
        
        return {
            "market_analysis": AgentOutput(
                content=market_result["market_analysis"], 
                timestamp=datetime.now(), 
                agent_id="marketresearch"
            ),
            "content_analysis": AgentOutput(
                content=content_result["content_analysis"], 
                timestamp=datetime.now(), 
                agent_id="content_analyzer"
            ),
            "current_step": "research_complete"
        }
    except Exception as e:
        if writer:
            writer({"status": f"Research team error: {str(e)}"})
        raise

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
    """Coordinates story arc, prose, and quality review teams."""
    arc_result = await story_arc_analyst_node(state, config=config)
    prose_result = await prosespecialist_node(state, config=config)
    quality_result = await quality_reviewer_node(state, config=config)
    
    return {
        "story_arc": AgentOutput(content=arc_result["story_arc"], timestamp=datetime.now(), agent_id="story_arc_analyst"),
        "prose": AgentOutput(content=prose_result["prose"], timestamp=datetime.now(), agent_id="prosespecialist"),
        "quality_review": AgentOutput(content=quality_result["quality_review"], timestamp=datetime.now(), agent_id="quality_reviewer"),
        "current_step": "complete"
    }

# Add individual agent node functions
async def marketresearch_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Market researcher agent node."""
    try:
        manuscript_state = state.get_manuscript_state()
        agent = MarketResearcher(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Market analysis returned no results")
        return {"market_analysis": result}
    except Exception as e:
        return {"error": f"Market research failed: {str(e)}"}

async def content_analyzer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Content analyzer agent node."""
    try:
        manuscript_state = state.get_manuscript_state()
        agent = ContentAnalyzer(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Content analysis returned no results")
        return {"content_analysis": result}
    except Exception as e:
        return {"error": f"Content analysis failed: {str(e)}"}

async def character_developer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = CharacterDeveloper(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Character development returned no results")
        return {"characters": result}
    except Exception as e:
        return {"error": f"Character development failed: {str(e)}"}

async def dialogue_enhancer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = DialogueEnhancer(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Dialogue enhancement returned no results")
        return {"dialogue": result}
    except Exception as e:
        return {"error": f"Dialogue enhancement failed: {str(e)}"}

async def world_builder_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = WorldBuilder(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("World building returned no results")
        return {"world_building": result}
    except Exception as e:
        return {"error": f"World building failed: {str(e)}"}

async def subplot_weaver_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = SubplotWeaver(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Subplot weaving returned no results")
        return {"subplots": result}
    except Exception as e:
        return {"error": f"Subplot weaving failed: {str(e)}"}

async def story_arc_analyst_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = StoryArcAnalyst(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Story arc analysis returned no results")
        return {"story_arc": result}
    except Exception as e:
        return {"error": f"Story arc analysis failed: {str(e)}"}

async def prosespecialist_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = ProseSpecialist(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Prose enhancement returned no results")
        return {"prose": result}
    except Exception as e:
        return {"error": f"Prose enhancement failed: {str(e)}"}

async def quality_reviewer_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    try:
        manuscript_state = state.get_manuscript_state()
        agent = QualityReviewer(config)
        result = await agent.process_manuscript(manuscript_state)
        if not result:
            raise ValueError("Quality review returned no results")
        return {"quality_review": result}
    except Exception as e:
        return {"error": f"Quality review failed: {str(e)}"}

def build_storybook(config: Optional[Dict[str, Any]] = None) -> LangChainGraph:
    """
    Build and return a story graph based on the provided configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LangChainGraph: The constructed story graph
    """
    # Get default config or use provided config
    config = config or get_default_config()
    
    # Ensure config is properly structured
    if not isinstance(config, dict):
        config = {"model": config} if isinstance(config, str) else {}
    
    # Merge with defaults if needed
    default_values = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    for key, value in default_values.items():
        if key not in config:
            config[key] = value

    # Initialize graph builder
    builder = StateGraph(State, input=InputState, config_schema=Configuration)
    
    # Add team workflow nodes
    team_structure = {
        "research_team_supervisor": ["marketresearch", "content_analyzer"],
        "creative_team_supervisor": ["character_developer", "dialogue_enhancer", "world_builder", "subplot_weaver"],
        "quality_team_supervisor": ["story_arc_analyst", "prosespecialist", "quality_reviewer"]
    }

    # Add all nodes
    for supervisor, agents in team_structure.items():
        builder.add_node(supervisor, globals()[supervisor])
        for agent in agents:
            agent_func = f"{agent}_node"
            builder.add_node(agent, globals()[agent_func])

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

    # Fixed interrupt node names to match actual supervisor nodes
    graph = builder.compile(
        checkpointer=MemorySaver(),
        debug=config.get("debug", False),
        interrupt_after=["research_team_supervisor", "creative_team_supervisor", "quality_team_supervisor"]
    )
    
    graph.name = "DetailedStoryBookGraph"
    graph.stream_mode = "updates"
    return graph

__all__ = ["build_storybook"]