from __future__ import annotations

from typing import Dict, Any, Optional, List, TypedDict
from langgraph.graph import StateGraph
from langgraph.channels import LastValue

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

class GraphState(TypedDict):
    """Type definition for graph state."""
    manuscript: Dict[str, Any]
    market_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    characters: List[Dict[str, Any]]
    dialogue: List[Dict[str, Any]]
    world_building: Dict[str, Any]
    subplots: List[Dict[str, Any]]
    story_arc: Dict[str, Any]
    language: Dict[str, Any]
    quality_review: Dict[str, Any]
    state: str

def build_storybook(manuscript_text: str, config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Build the storybook processing graph."""
    workflow = StateGraph()
    
    # Initialize agents
    agents = {
        "market_researcher": MarketResearcher(config),
        "content_analyzer": ContentAnalyzer(config),
        "character_developer": CharacterDeveloper(config),
        "dialogue_enhancer": DialogueEnhancer(config),
        "world_builder": WorldBuilder(config),
        "subplot_weaver": SubplotWeaver(config),
        "story_arc_analyst": StoryArcAnalyst(config),
        "language_polisher": LanguagePolisher(config),
        "quality_reviewer": QualityReviewer(config)
    }

    # Add state channels
    workflow.add_state("manuscript", LastValue[Dict[str, Any]])
    workflow.add_state("market_analysis", LastValue[Dict[str, Any]])
    workflow.add_state("content_analysis", LastValue[Dict[str, Any]])
    workflow.add_state("characters", LastValue[List[Dict[str, Any]]])
    workflow.add_state("dialogue", LastValue[List[Dict[str, Any]]])
    workflow.add_state("world_building", LastValue[Dict[str, Any]])
    workflow.add_state("subplots", LastValue[List[Dict[str, Any]]])
    workflow.add_state("story_arc", LastValue[Dict[str, Any]])
    workflow.add_state("language", LastValue[Dict[str, Any]])
    workflow.add_state("quality_review", LastValue[Dict[str, Any]])
    workflow.add_state("state", LastValue[str])

    @workflow.node
    async def market_research(state):
        """Analyze market positioning and trends."""
        results = await agents["market_researcher"].process_manuscript(state["manuscript"])
        return {
            "market_analysis": results,
            "state": "market_analyzed"
        }

    @workflow.node
    async def content_analysis(state):
        """Analyze content and themes."""
        results = await agents["content_analyzer"].process_manuscript(state["manuscript"])
        return {
            "content_analysis": results,
            "state": "content_analyzed"
        }

    @workflow.node
    async def creative_development(state):
        """Develop creative elements."""
        character_results = await agents["character_developer"].process_manuscript(
            state["manuscript"]
        )
        dialogue_results = await agents["dialogue_enhancer"].process_manuscript(
            state["manuscript"],
            characters=character_results
        )
        world_results = await agents["world_builder"].process_manuscript(
            state["manuscript"]
        )
        subplot_results = await agents["subplot_weaver"].process_manuscript(
            state["manuscript"],
            characters=character_results
        )
        
        return {
            "characters": character_results,
            "dialogue": dialogue_results,
            "world_building": world_results,
            "subplots": subplot_results,
            "state": "creative_complete"
        }

    @workflow.node
    async def story_development(state):
        """Develop story structure and language."""
        arc_results = await agents["story_arc_analyst"].process_manuscript(
            state["manuscript"],
            characters=state["characters"],
            subplots=state["subplots"]
        )
        language_results = await agents["language_polisher"].process_manuscript(
            state["manuscript"]
        )
        
        return {
            "story_arc": arc_results,
            "language": language_results,
            "state": "story_complete"
        }

    @workflow.node
    async def quality_review(state):
        """Review and validate story elements."""
        review_results = await agents["quality_reviewer"].process_manuscript(
            state["manuscript"],
            context=state
        )
        return {
            "quality_review": review_results,
            "state": "complete"
        }

    # Configure workflow routing
    workflow.set_entry_point("market_research")
    workflow.add_edge("market_research", "content_analysis")
    workflow.add_edge("content_analysis", "creative_development")
    workflow.add_edge("creative_development", "story_development")
    workflow.add_edge("story_development", "quality_review")
    
    # Initialize state
    workflow.set_initial_state({
        "manuscript": {"text": manuscript_text},
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
    })
    
    return workflow