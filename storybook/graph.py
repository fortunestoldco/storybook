from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel  # Updated import

from storybook.state import State, InputState, AgentOutput
from storybook.agents import (
    MarketResearcher, ContentAnalyzer, CharacterDeveloper,
    DialogueEnhancer, WorldBuilder, SubplotWeaver,
    StoryArcAnalyst, LanguagePolisher, QualityReviewer
)
from storybook.config import Configuration

# Node functions
async def research_market(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Market research node."""
    agent = MarketResearcher(config)
    result = await agent.process_manuscript(state.manuscript)
    return {
        "market_analysis": AgentOutput(
            content=result,
            timestamp=datetime.now(),
            agent_id="market_researcher"
        ),
        "current_step": "market_analyzed"
    }

async def analyze_content(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Content analysis node."""
    agent = ContentAnalyzer(config)
    result = await agent.process_manuscript(state.manuscript)
    return {
        "content_analysis": AgentOutput(
            content=result,
            timestamp=datetime.now(),
            agent_id="content_analyzer"
        ),
        "current_step": "content_analyzed"
    }

async def develop_creative(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Creative development node."""
    char_agent = CharacterDeveloper(config)
    dialog_agent = DialogueEnhancer(config)
    world_agent = WorldBuilder(config)
    subplot_agent = SubplotWeaver(config)
    
    char_result = await char_agent.process_manuscript(state.manuscript)
    dialog_result = await dialog_agent.process_manuscript(state.manuscript)
    world_result = await world_agent.process_manuscript(state.manuscript)
    subplot_result = await subplot_agent.process_manuscript(state.manuscript)
    
    return {
        "world_building": AgentOutput(content=world_result, timestamp=datetime.now(), agent_id="world_builder"),
        "subplots": AgentOutput(content=subplot_result, timestamp=datetime.now(), agent_id="subplot_weaver"),
        "current_step": "creative_complete"
    }

async def story_development(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Story development node."""
    arc_agent = StoryArcAnalyst(config)
    lang_agent = LanguagePolisher(config)
    
    arc_result = await arc_agent.process_manuscript(state.manuscript)
    lang_result = await lang_agent.process_manuscript(state.manuscript)
    
    return {
        "story_arc": AgentOutput(content=arc_result, timestamp=datetime.now(), agent_id="story_arc_analyst"),
        "language": AgentOutput(content=lang_result, timestamp=datetime.now(), agent_id="language_polisher"),
        "current_step": "story_complete"
    }

async def quality_review(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Quality review node."""
    agent = QualityReviewer(config)
    result = await agent.process_manuscript(state.manuscript)
    return {
        "quality_review": AgentOutput(
            content=result,
            timestamp=datetime.now(),
            agent_id="quality_reviewer"
        ),
        "current_step": "complete"
    }

# Build graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("market_research", market_research)
builder.add_node("content_analysis", content_analysis)
builder.add_node("creative_development", creative_development)
builder.add_node("story_development", story_development)
builder.add_node("quality_review", quality_review)

# Add edges
builder.add_edge("__start__", "market_research")
builder.add_edge("market_research", "content_analysis")
builder.add_edge("content_analysis", "creative_development")
builder.add_edge("creative_development", "story_development")
builder.add_edge("story_development", "quality_review")

# Compile graph
graph = builder.compile()
graph.name = "StoryBookGraph"