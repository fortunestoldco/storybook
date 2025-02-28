from __future__ import annotations

from typing import Dict, Any, Optional, Annotated, List
import os
import logging
from pathlib import Path

from langchain.schema import Document
from langgraph.graph import Graph, StateType
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

def build_storybook(config: Optional[Dict[str, Any]] = None) -> Graph:
    """Build the storybook workflow graph."""
    
    # Define state type with Annotated for multi-value channels
    class State(StateType):
        manuscript: Dict[str, Any]
        characters: Annotated[List[Dict[str, Any]], "characters"]
        research: Annotated[Dict[str, Any], "research"]
        analysis: Annotated[Dict[str, Any]], "analysis"]
        improvements: Annotated[List[Dict[str, Any]], "improvements"]

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
    def init_state(state):
        return {"state": STATES["START"], "manuscript": state["manuscript"]}

    workflow.add_node("start", init_state)

    # Add agent nodes with proper state updates
    for state in STATES.values():
        if state not in ["start", "end"]:
            agent = agents[state.lower().replace("_", "")]
            workflow.add_node(state, lambda x, agent=agent: agent.process(x))

    workflow.add_node("end", lambda x: {"state": STATES["END"], **x})

    # Define edges with conditional routing
    edges = [
        ("start", "research"),
        ("research", "analysis"),
        ("analysis", "initialize"),
        ("initialize", "character_development"),
        ("character_development", "dialogue_enhancement"),
        ("dialogue_enhancement", "world_building"),
        ("world_building", "subplot_integration"),
        ("subplot_integration", "story_arc_evaluation"),
        ("story_arc_evaluation", "continuity_check"),
        ("continuity_check", "language_polishing"),
        ("language_polishing", "quality_review"),
        ("quality_review", "finalize"),
        ("finalize", "end")
    ]

    # Add edges with conditions
    for edge in edges:
        workflow.add_edge(edge[0], edge[1])

    return workflow
