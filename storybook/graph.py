from __future__ import annotations

from typing import Dict, Any, List, TypedDict
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

# Schema Definitions
class InputState(TypedDict):
    """Input state schema definition."""
    manuscript_text: str

class OutputState(TypedDict):
    """Output state schema definition."""
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
    """Overall graph state schema."""
    state: str

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build the storybook processing graph."""
    # Initialize workflow with state schemas
    workflow = StateGraph(
        GraphState,
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

    async def market_research(state: GraphState):
        """Analyze market positioning and trends."""
        results = await agents["market_researcher"].process_manuscript(state)
        return {
            "market_analysis": results,
            "state": "market_analyzed"
        }

    async def content_analysis(state: GraphState):
        """Analyze content and themes."""
        results = await agents["content_analyzer"].process_manuscript(state)
        return {
            "content_analysis": results,
            "state": "content_analyzed"
        }

    async def creative_development(state: GraphState):
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

    async def story_development(state: GraphState):
        """Develop story structure and language."""
        arc_results = await agents["story_arc_analyst"].process_manuscript(
            state,
            characters=state["characters"],
            subplots=state["subplots"]
        )
        language_results = await agents["language_polisher"].process_manuscript(state)
        
        return {
            "story_arc": arc_results,
            "language": language_results,
            "state": "story_complete"
        }

    async def quality_review(state: GraphState):
        """Review and validate story elements."""
        review_results = await agents["quality_reviewer"].process_manuscript(
            state,
            context=state
        )
        return {
            "quality_review": review_results,
            "state": "complete"
        }

    # Add nodes to workflow
    workflow.add_node("market_research", market_research)
    workflow.add_node("content_analysis", content_analysis)
    workflow.add_node("creative_development", creative_development)
    workflow.add_node("story_development", story_development)
    workflow.add_node("quality_review", quality_review)

    # Configure workflow routing
    workflow.add_edge(START, "market_research")
    workflow.add_edge("market_research", "content_analysis")
    workflow.add_edge("content_analysis", "creative_development")
    workflow.add_edge("creative_development", "story_development")
    workflow.add_edge("story_development", "quality_review")
    workflow.add_edge("quality_review", END)

    # Initialize state
    initial_state: GraphState = {
        "manuscript_text": "",  # Will be set by input
        "market_analysis": {},
        "content_analysis": {},
        "characters": [],
        "dialogue": [],
        "world_building": {},
        "subplots": [],
        "story_arc": {},
        "language": {},
        "quality_review": {},
        "state": "start"
    }
    
    workflow.set_initial_state(initial_state)
    
    return workflow.compile()
