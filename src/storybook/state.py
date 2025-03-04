"""Define the state structures for the storybook system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class ProjectInfo:
    """Information about the novel project."""
    
    title: str = ""
    genre: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    length_target: Dict[str, Any] = field(default_factory=dict)
    content_guidelines: Dict[str, Any] = field(default_factory=dict)
    timeline: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputState:
    """Defines the input state for the storybook system."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """Messages tracking the primary execution state of the system."""
    
    project_id: str = ""
    """Unique identifier for the novel project."""
    
    phase: str = "initialization"
    """Current phase of the novel writing process."""
    
    task: str = ""
    """Current task being processed."""


@dataclass
class NovelSystemState(InputState):
    """Represents the complete state of the storybook system."""

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one before the graph raises an error."""
    
    project: ProjectInfo = field(default_factory=ProjectInfo)
    """Information about the current novel project."""
    
    current_input: Dict[str, Any] = field(default_factory=dict)
    """Current input being processed by an agent."""
    
    current_agent: str = ""
    """The current agent processing the input."""
    
    agent_outputs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    """Outputs from various agents, organized by agent name."""
    
    phase_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    """Historical record of activities in each phase."""
