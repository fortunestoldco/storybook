from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

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


class InputState(BaseModel):
    """
    Input state schema definition.
    """
    manuscript_text: str


class OutputState(BaseModel):
    """
    Output state schema definition.
    """
    market_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    characters: List[Dict[str, Any]]
    dialogue: List[Dict[str, Any]]
    world_building: Dict[str, Any]
    subplots: List[Dict[str, Any]]
    story_arc: Dict[str, Any]
    language: Dict[str, Any]
    quality_review: Dict[str, Any]


class GraphState(InputState, OutputState):
    """
    Overall graph state schema.
    Inherits fields from both InputState and OutputState.
    """
    state: str


def build_storybook(config: RunnableConfig) -> StateGraph:
    """
    Build the storybook processing graph.
    Replaces TypedDict with Pydantic models for better compatibility
    with langgraph's configuration schema logic.
    """
    # Initialize workflow with state schemas and default state
    initial_state = GraphState(
        manuscript_text="",
        market_analysis={},
        content_analysis={},
        characters=[],
        dialogue=[],
        world_building={},
        subplots=[],
        story_arc={},
        language={},
        quality_review={},
        state="start"
    )

    workflow = StateGraph(
        GraphState,
        initial_state,
        input=InputState,
        output=OutputState
    )

    # Get LLM configuration
    llm_config = get_llm(config.get("metadata", {}))

    # Initialize agents with config
    agents = {
        "market_researcher": MarketResearcher(llm_config),
        "content_analyzer": ContentAnalyzer(llm_config),
        "character_developer": CharacterDeveloper(llm_config),
        "dialogue_enhancer": DialogueEnhancer(llm_config),
        "world_builder": WorldBuilder(llm_config),
        "subplot_weaver": SubplotWeaver(llm_config),
        "story_arc_analyst": StoryArcAnalyst(llm_config),
        "language_polisher": LanguagePolisher(llm_config),
        "quality_reviewer": QualityReviewer(llm_config)
    }

    async def market_research(state: GraphState) -> Dict[str, Any]:
        """Analyze market positioning and trends."""
        results = await agents["market_researcher"].process_manuscript(state)
        return {
            "market_analysis": results,
            "state": "market_analyzed"
        }

    async def content_analysis(state: GraphState) -> Dict[str, Any]:
        """Analyze content and themes."""
        results = await agents["content_analyzer"].process_manuscript(state)
        return {
            "content_analysis": results,
            "state": "content_analyzed"
        }

    async def creative_development(state: GraphState) -> Dict[str, Any]:
        """Develop creative elements."""
        character_results = await agents["character_developer"].process_manuscript(state)
        dialogue_results = await agents["dialogue_enhancer"].process_manuscript(
            state,
            characters=character_results
        )
        world_results = await agents["world_builder"].process_manuscript(state)
        subplot_results = await agents["subplot_weaver"].process_manuscript(
            state,
            characters=character_results
        )

        return {
            "characters": character_results,
            "dialogue": dialogue_results,
            "world_building": world_results,
            "subplots": subplot_results,
            "state": "creative_complete"
        }

    async def story_development(state: GraphState) -> Dict[str, Any]:
        """Develop story structure and language."""
        arc_results = await agents["story_arc_analyst"].process_manuscript(
            state,
            characters=state.characters,
            subplots=state.subplots
        )
        language_results = await agents["language_polisher"].process_manuscript(state)

        return {
            "story_arc": arc_results,
            "language": language_results,
            "state": "story_complete"
        }

    async def quality_review(state: GraphState) -> Dict[str, Any]:
        """Review and validate story elements."""
        review_results = await agents["quality_reviewer"].process_manuscript(
            state,
            context=state.dict()
        )
        return {
            "quality_review": review_results,
            "state": "complete"
        }

    # Add nodes to workflow
    workflow.add_node("analyze_market", market_research)
    workflow.add_node("analyze_content", content_analysis)
    workflow.add_node("develop_creative", creative_development)
    workflow.add_node("develop_story", story_development)
    workflow.add_node("review_quality", quality_review)

    # Configure workflow routing
    workflow.add_edge(START, "analyze_market")
    workflow.add_edge("analyze_market", "analyze_content")
    workflow.add_edge("analyze_content", "develop_creative")
    workflow.add_edge("develop_creative", "develop_story")
    workflow.add_edge("develop_story", "review_quality")
    workflow.add_edge("review_quality", END)

    return workflow.compile()
