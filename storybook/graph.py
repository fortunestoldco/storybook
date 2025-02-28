from __future__ import annotations

from typing import Dict, Any, Optional, Annotated, List, TypedDict
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
    
    channels = {
        "manuscript": LastValue(Dict[str, Any]),  # Specify type as Dict[str, Any]
        "characters": LastValue(List[Dict[str, Any]]),  # Changed from default parameter
        "research": LastValue(default={}),
        "analysis": LastValue(default={}),
        "improvements": LastValue(default=[]),
        "status": LastValue(default=STATES["START"])
    }

    # Add channels to graph
    workflow = StateGraph()
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

    # Add start node that initializes state
    def init_state(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "manuscript": inputs.get("manuscript", {}),
            "characters": [],
            "research": {},
            "analysis": {},
            "improvements": [],
            "status": STATES["START"]
        }

    workflow.add_node("start", init_state)

    # Add agent nodes
    for state_name in STATES:
        if state_name not in ["START", "END"]:
            agent_name = state_name.lower().replace("_", "")
            if agent_name in agents:
                def create_agent_node(agent):
                    def agent_node(inputs: Dict[str, Any]) -> Dict[str, Any]:
                        result = agent.process(inputs)
                        return {**inputs, **result}
                    return agent_node
                
                workflow.add_node(state_name, create_agent_node(agents[agent_name]))

    # Add end node that finalizes state
    def finalize_state(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {**inputs, "status": STATES["END"]}

    workflow.add_node("end", finalize_state)

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

    return workflow.compile()
