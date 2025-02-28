from __future__ import annotations

from typing import Dict, Any, Optional, Annotated, List, TypedDict
import os
import logging
from pathlib import Path

from langchain.schema import Document
from langgraph.graph import Graph
from langgraph.channels import LastValue
from langgraph.pregel import channels  # Changed from BaseChannel
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
    
    workflow = Graph()

    # Create channels for state management using correct imports
    channels = {
        "manuscript": LastValue(),
        "characters": LastValue(default=[]),
        "research": LastValue(default={}),
        "analysis": LastValue(default={}),
        "improvements": LastValue(default=[]),
        "status": LastValue(default=STATES["START"])
    }

    # Add channels to graph
    for name, channel in channels.items():
        workflow.add_channel(name, channel)

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
    workflow.add_node("start", lambda x: x)

    # Add agent nodes
    for state_name in STATES:
        if state_name not in ["START", "END"]:
            agent_name = state_name.lower().replace("_", "")
            if agent_name in agents:
                workflow.add_node(
                    state_name,
                    lambda x, agent=agents[agent_name]: agent.process(x)
                )

    workflow.add_node("end", lambda x: {**x, "status": STATES["END"]})

    # Define and add edges
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

    for start, end in edges:
        workflow.add_edge(start, end)

    return workflow.compile()
