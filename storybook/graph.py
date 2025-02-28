from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

from storybook.state import State, InputState
from storybook.agents import (
    MarketResearcher, ContentAnalyzer, CharacterDeveloper,
    DialogueEnhancer, WorldBuilder, SubplotWeaver,
    StoryArcAnalyst, LanguagePolisher, QualityReviewer
)
from storybook.config import Configuration

async def market_research(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
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