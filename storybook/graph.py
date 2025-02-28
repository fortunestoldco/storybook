from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

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
    manuscript_state = ManuscriptState(
        title=state.title,
        manuscript=state.manuscript,
        notes=state.notes,
        llm_provider=state.llm_provider
    )
    
    state.manuscript = manuscript_state
    result = await agent.process_manuscript(manuscript_state)
    
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
        "characters": AgentOutput(content=char_result, timestamp=datetime.now(), agent_id="character_developer"),
        "dialogue": AgentOutput(content=dialog_result, timestamp=datetime.now(), agent_id="dialogue_enhancer"),
        "world_building": AgentOutput(content=world_result, timestamp=datetime.now(), agent_id="world_builder"),
        "subplots": AgentOutput(content=subplot_result, timestamp=datetime.now(), agent_id="subplot_weaver"),
        "current_step": "creative_complete"
    }

async def develop_story(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
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

async def review_quality(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
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

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build and return the storybook processing graph."""
    # Initialize workflow with new input schema
    builder = StateGraph(State, input=InputState, config_schema=Configuration)

    # Add nodes
    builder.add_node("research_market", research_market)
    builder.add_node("analyze_content", analyze_content)
    builder.add_node("develop_creative", develop_creative)
    builder.add_node("develop_story", develop_story)
    builder.add_node("review_quality", review_quality)

    # Add edges
    builder.add_edge("__start__", "research_market")
    builder.add_edge("research_market", "analyze_content")
    builder.add_edge("analyze_content", "develop_creative")
    builder.add_edge("develop_creative", "develop_story")
    builder.add_edge("develop_story", "review_quality")

    # Compile and return graph
    graph = builder.compile()
    graph.name = "StoryBookGraph"
    return graph

# Export the graph builder function
__all__ = ["build_storybook"]