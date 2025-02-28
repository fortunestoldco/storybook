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

async def research_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates and combines market and content analysis."""
    market_agent = MarketResearcher(config)
    content_agent = ContentAnalyzer(config)
    
    # Run analyses in parallel
    market_result = await market_agent.process_manuscript(state.manuscript)
    content_result = await content_agent.process_manuscript(state.manuscript)
    
    return {
        "market_analysis": AgentOutput(
            content=market_result,
            timestamp=datetime.now(),
            agent_id="market_researcher"
        ),
        "content_analysis": AgentOutput(
            content=content_result,
            timestamp=datetime.now(),
            agent_id="content_analyzer"
        ),
        "current_step": "research_complete"
    }

async def creative_team_supervisor(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Coordinates character, dialogue, world-building, and subplot teams."""
    char_agent = CharacterDeveloper(config)
    dialog_agent = DialogueEnhancer(config)
    world_agent = WorldBuilder(config)
    subplot_agent = SubplotWeaver(config)
    
    # Run creative development in parallel
    results = await asyncio.gather(
        char_agent.process_manuscript(state.manuscript),
        dialog_agent.process_manuscript(state.manuscript),
        world_agent.process_manuscript(state.manuscript),
        subplot_agent.process_manuscript(state.manuscript)
    )
    
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
    
    # Run quality checks in parallel
    arc_result, lang_result = await asyncio.gather(
        arc_agent.process_manuscript(state.manuscript),
        lang_agent.process_manuscript(state.manuscript)
    )
    
    # Final quality review after improvements
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

    # Add supervisor nodes
    builder.add_node("research_team", research_team_supervisor)
    builder.add_node("creative_team", creative_team_supervisor)
    builder.add_node("quality_team", quality_team_supervisor)

    # Define the hierarchical workflow
    builder.add_edge("__start__", "research_team")
    builder.add_edge("research_team", "creative_team")
    builder.add_edge("creative_team", "quality_team")

    # Add conditional routing
    def should_revise(state: State) -> bool:
        return state.quality_review.content.get("needs_revision", False)

    builder.add_conditional_edges(
        "quality_team",
        condition=should_revise,
        if_true="creative_team",  # Loop back for revision
        if_false="__end__"  # Complete the process
    )

    # Compile and return graph
    graph = builder.compile()
    graph.name = "HierarchicalStoryBookGraph"
    return graph

# Export the graph builder function
__all__ = ["build_storybook"]
