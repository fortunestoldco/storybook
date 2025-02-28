from __future__ import annotations

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from storybook.config import create_llm, get_llm
from storybook.agents import (
    CharacterDeveloper,
    DialogueEnhancer,
    WorldBuilder,
    SubplotWeaver,
    StoryArcAnalyst,
    ContinuityEditor,
    LanguagePolisher,
    QualityReviewer,
    MarketResearcher,
    ContentAnalyzer
)
from storybook.state import State, InputState
from storybook.config import Configuration

# Input schema definition
class InputState(TypedDict):
    manuscript_text: str

# Output schema definition
class OutputState(TypedDict):
    market_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    characters: List[Dict[str, Any]]
    dialogue: List[Dict[str, Any]]
    world_building: Dict[str, Any]
    subplots: List[Dict[str, Any]]
    story_arc: Dict[str, Any]
    language: Dict[str, Any]
    quality_review: Dict[str, Any]

# Overall state schema
class GraphState(InputState, OutputState):
    """Type definition for graph state."""
    state: str

async def market_research(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Market research node."""
    agent = MarketResearcher(config)
    result = await agent.process_manuscript(state.manuscript)
    return {
        "market_analysis": {
            "content": result,
            "timestamp": datetime.now(),
            "agent_id": "market_researcher"
        },
        "current_step": "market_analyzed"
    }

# ... similar implementations for other nodes ...

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