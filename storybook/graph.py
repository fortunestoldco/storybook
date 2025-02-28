from __future__ import annotations

from typing import Dict, Any, Optional, Annotated, List, TypedDict
import os
import logging
from pathlib import Path

from langchain.schema import Document
from langgraph.graph import Graph, END, StateType
from langgraph.prebuilt.tool_executor import ToolExecutor
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

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """Type definition for graph state."""
    manuscript: Dict[str, Any]
    characters: List[Dict[str, Any]]
    research: Dict[str, Any]
    analysis: Dict[str, Any]
    improvements: List[Dict[str, Any]]
    status: str

def build_storybook(config: Optional[Dict[str, Any]] = None) -> Graph:
    """Build the storybook workflow graph."""
    
    # Initialize tools and agents
    agents = {
        "character_developer": CharacterDeveloper(config),
        "dialogue_enhancer": DialogueEnhancer(config),
        "world_builder": WorldBuilder(config),
        "subplot_weaver": SubplotWeaver(config),
        "story_arc_analyst": StoryArcAnalyst(config),
        "continuity_editor": ContinuityEditor(config),
        "language_polisher": LanguagePolisher(config),
        "quality_reviewer": QualityReviewer(config),
        "market_researcher": MarketResearcher(config),
        "content_analyzer": ContentAnalyzer(config)
    }

    # Create the workflow graph
    workflow = Graph()

    # Add nodes with proper state handling
    def init_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the graph state."""
        return {
            "manuscript": state.get("manuscript", {}),
            "characters": [],
            "research": {},
            "analysis": {},
            "improvements": [],
            "status": STATES["START"]
        }

    workflow.add_node("start", init_state)

    # Add agent nodes with proper state updates
    for state_name in STATES:
        if state_name not in ["START", "END"]:
            agent_name = state_name.lower().replace("_", "")
            if agent_name in agents:
                workflow.add_node(
                    state_name,
                    lambda x, agent=agents[agent_name]: agent.process(x)
                )

    workflow.add_node("end", lambda x: {**x, "status": STATES["END"]})

    # Define edges with conditional routing
    edges = [
        ("start", "RESEARCH"),
        ("RESEARCH", "ANALYSIS"),
        ("ANALYSIS", "CHARACTER_DEVELOPMENT"),
        ("CHARACTER_DEVELOPMENT", "DIALOGUE_ENHANCEMENT"),
        ("DIALOGUE_ENHANCEMENT", "WORLD_BUILDING"),
        ("WORLD_BUILDING", "SUBPLOT_INTEGRATION"),
        ("SUBPLOT_INTEGRATION", "STORY_ARC_EVALUATION"),
        ("STORY_ARC_EVALUATION", "CONTINUITY_CHECK"),
        ("CONTINUITY_CHECK", "LANGUAGE_POLISHING"),
        ("LANGUAGE_POLISHING", "QUALITY_REVIEW"),
        ("QUALITY_REVIEW", "end")
    ]

    # Add edges to graph
    for start, end in edges:
        workflow.add_edge(start, end)

    return workflow
