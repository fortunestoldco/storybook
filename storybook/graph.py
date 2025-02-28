from __future__ import annotations

from typing import Dict, Any, Optional, List, TypedDict
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

def build_storybook(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Build the storybook processing graph."""
    
    # Create workflow graph
    workflow = StateGraph()
    
    # Add channels
    workflow.add_channel("manuscript", LastValue(Dict[str, Any]))
    workflow.add_channel("characters", LastValue(List[Dict[str, Any]]))
    workflow.add_channel("subplots", LastValue(List[Dict[str, Any]]))
    workflow.add_channel("themes", LastValue(List[Dict[str, Any]]))
    workflow.add_channel("world_building", LastValue(Dict[str, Any]))
    workflow.add_channel("state", LastValue(str))

    # Initialize agents
    character_analyst = CharacterAnalyst()
    subplot_weaver = SubplotWeaver()
    theme_enhancer = ThemeEnhancer()
    pacing_editor = PacingEditor()
    world_builder = WorldBuilder()

    @workflow.node
    def analyze_characters(state):
        manuscript = state["manuscript"]
        results = character_analyst.process_manuscript(manuscript["id"])
        return {"characters": results}

    @workflow.node
    def develop_subplots(state):
        manuscript = state["manuscript"]
        characters = state["characters"]
        results = subplot_weaver.process_manuscript(
            manuscript["id"], 
            characters=characters
        )
        return {"subplots": results}

    @workflow.node
    def enhance_themes(state):
        manuscript = state["manuscript"]
        characters = state["characters"]
        subplots = state["subplots"]
        results = theme_enhancer.process_manuscript(
            manuscript["id"],
            characters=characters,
            subplots=subplots
        )
        return {"themes": results}

    @workflow.node
    def build_world(state):
        manuscript = state["manuscript"]
        results = world_builder.process_manuscript(manuscript["id"])
        return {"world_building": results}

    @workflow.node
    def edit_pacing(state):
        manuscript = state["manuscript"]
        subplots = state["subplots"]
        results = pacing_editor.process_manuscript(
            manuscript["id"],
            subplots=subplots
        )
        return {"manuscript": results}

    # Set up graph structure
    workflow.set_entry_point("analyze_characters")
    workflow.add_edge("analyze_characters", "develop_subplots")
    workflow.add_edge("develop_subplots", "enhance_themes")
    workflow.add_edge("enhance_themes", "build_world")
    workflow.add_edge("build_world", "edit_pacing")
    
    # Initialize state
    workflow.set_initial_state({
        "manuscript": {},
        "characters": [],
        "subplots": [],
        "themes": [],
        "world_building": {},
        "state": "start"
    })

    return workflow
