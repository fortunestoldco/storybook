from __future__ import annotations

from typing import Dict, Any, Optional, Annotated, List, TypedDict
import os
import logging
from pathlib import Path

from langchain.schema import Document
from langgraph.graph import Graph
from langgraph.channels import LastValue
from langgraph_core.state import StateGraph
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

def build_storybook(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Build the storybook workflow graph."""
    
    # Create state graph with defined state type
    workflow = StateGraph(GraphState)

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

    # Add nodes with proper state handling
    def init_state(state: Dict[str, Any]) -> GraphState:
        """Initialize the graph state."""
        return GraphState(
            manuscript=state.get("manuscript", {}),
            characters=[],
            research={},
            analysis={},
            improvements=[],
            status=STATES["START"]
        )

    # Add start node
    workflow.add_node("start", init_state)

    # Add agent nodes
    for state_name in STATES:
        if state_name not in ["START", "END"]:
            agent_name = state_name.lower().replace("_", "")
            if agent_name in agents:
                workflow.add_node(
                    state_name,
                    lambda x, agent=agents[agent_name]: agent.process(x)
                )

    # Add end node
    workflow.add_node("end", lambda x: GraphState(**{**x, "status": STATES["END"]}))

    # Define edges
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

    # Add edges
    for start, end in edges:
        workflow.add_edge(start, end)

    # Compile the graph
    return workflow.compile()
