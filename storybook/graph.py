from __future__ import annotations

from typing import Dict, Any, Optional, List, TypedDict, Literal
import os
import logging
from pathlib import Path

from langchain.schema import Document
from langgraph.graph import StateGraph
from langgraph.channels import LastValue
from langchain_core.tools import BaseTool

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
from storybook.config import validate_agent_config, STATES

from .agents.character_analyst import CharacterAnalyst
from .agents.subplot_weaver import SubplotWeaver
from .agents.theme_enhancer import ThemeEnhancer
from .agents.pacing_editor import PacingEditor
from .agents.world_builder import WorldBuilder

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """Type definition for graph state."""
    manuscript: Dict[str, Any]
    characters: List[Dict[str, Any]]
    research: Dict[str, Any]
    analysis: Dict[str, Any]
    improvements: List[Dict[str, Any]]
    status: str

def next_step(state: Dict[str, Any]) -> Literal[
    "project_setup",
    "market_analysis", 
    "story_planning",
    "character_development",
    "world_building",
    "subplot_development",
    "end"
]:
    """Route to next step based on state."""
    current_state = state["state"]
    
    if current_state == "start":
        return "project_setup"
    elif current_state == "setup_complete":
        return "market_analysis"
    elif current_state == "market_analyzed":
        return "story_planning"
    elif current_state == "story_planned":
        return "character_development"
    elif current_state == "characters_developed":
        return "world_building"
    elif current_state == "world_built":
        return "subplot_development"
    else:
        return "end"

def build_storybook(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Build the storybook processing graph."""
    
    workflow = StateGraph()
    
    # Add channels
    workflow.add_channel("manuscript", LastValue(Dict[str, Any]))
    workflow.add_channel("market_analysis", LastValue(Dict[str, Any]))
    workflow.add_channel("characters", LastValue(List[Dict[str, Any]]))
    workflow.add_channel("subplots", LastValue(List[Dict[str, Any]]))
    workflow.add_channel("world_building", LastValue(Dict[str, Any]))
    workflow.add_channel("state", LastValue(str))

    @workflow.node
    def project_setup(state):
        """Initialize project settings and baseline manuscript."""
        return {
            "manuscript": {"id": state.get("manuscript_id"), "content": ""},
            "state": "setup_complete"
        }

    @workflow.node
    def market_analysis(state):
        """Analyze market trends and target audience."""
        return {
            "market_analysis": {"target_audience": {}, "genre_requirements": {}},
            "state": "market_analyzed"
        }

    @workflow.node
    def story_planning(state):
        """Plan core story elements."""
        return {
            "manuscript": {**state["manuscript"], "plot_outline": []},
            "state": "story_planned"
        }

    @workflow.node
    def character_development(state):
        """Develop and analyze characters."""
        character_results = state["character_analyst"].process_manuscript(
            state["manuscript"]["id"]
        ) if "character_analyst" in state else []
        
        return {
            "characters": character_results,
            "state": "characters_developed"
        }

    @workflow.node
    def world_building(state):
        """Build and enrich story world."""
        world_results = state["world_builder"].process_manuscript(
            state["manuscript"]["id"]
        ) if "world_builder" in state else {}
        
        return {
            "world_building": world_results,
            "state": "world_built"
        }

    @workflow.node
    def subplot_development(state):
        """Develop and integrate subplots."""
        subplot_results = state["subplot_weaver"].process_manuscript(
            state["manuscript"]["id"],
            state.get("characters", [])
        ) if "subplot_weaver" in state else []
        
        return {
            "subplots": subplot_results,
            "state": "complete"
        }

    # Configure workflow routing
    workflow.set_entry_point("project_setup")
    workflow.add_conditional_edges(next_step)
    
    # Initialize state
    workflow.set_initial_state({
        "manuscript": {},
        "market_analysis": {},
        "characters": [],
        "subplots": [],
        "world_building": {},
        "state": "start"
    })

    return workflow
