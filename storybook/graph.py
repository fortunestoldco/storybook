from __future__ import annotations

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from storybook.config import get_llm
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
    
    # Initialize agents with LLM config
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

    # ... rest of the node implementations ...

    # Configure workflow routing
    workflow.add_edge(START, "market_research")
    workflow.add_edge("market_research", "content_analysis")
    workflow.add_edge("content_analysis", "creative_development")
    workflow.add_edge("creative_development", "story_development")
    workflow.add_edge("story_development", "quality_review")
    workflow.add_edge("quality_review", END)

    return workflow.compile()
