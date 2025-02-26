from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Status of the novel project."""

    INITIALIZED = "initialized"
    RESEARCHING = "researching"
    PLANNING = "planning"
    CHARACTER_DEVELOPMENT = "character_development"
    DRAFTING = "drafting"
    REVISING = "revising"
    OPTIMIZING = "optimizing"
    PREPARING_PUBLICATION = "preparing_publication"
    COMPLETED = "completed"


class ProjectConcept(BaseModel):
    """Model representing the core concept of the novel project."""

    premise: str
    genre: str
    target_audience: str
    theme: str
    setting: Optional[str] = None
    hook: Optional[str] = None
    estimated_length: Optional[int] = None


class ResearchItem(BaseModel):
    """A research item collected during the research phase."""

    topic: str
    content: str
    sources: List[str]
    relevance_score: float = 1.0
    verified: bool = False


class Character(BaseModel):
    """Character information."""

    name: str
    role: str
    background: str
    personality: Dict[str, Any]
    motivations: List[str]
    arc: Dict[str, Any]
    relationships: Dict[str, str]
    dialogue_patterns: Dict[str, Any]
    tropes: List[str]
    trope_subversions: List[str]


class PlotPoint(BaseModel):
    """A significant point in the plot."""

    title: str
    description: str
    characters_involved: List[str]
    chapter: Optional[int] = None
    tension_level: float = 0.5
    resolution_status: bool = False


class Chapter(BaseModel):
    """A chapter in the novel."""

    number: int
    title: str
    pov_character: Optional[str] = None
    summary: str
    content: str = ""
    word_count: int = 0
    completed: bool = False
    revision_count: int = 0
    quality_metrics: Dict[str, float] = Field(default_factory=dict)


class NovelState(BaseModel):
    """The complete state of the novel generation process."""

    # Project metadata
    project_id: str
    project_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: ProjectStatus = ProjectStatus.INITIALIZED

    # Project details
    genre: str
    subgenres: List[str] = Field(default_factory=list)
    target_audience: str
    target_word_count: int
    current_word_count: int = 0

    # Content components
    premise: str = ""
    themes: List[str] = Field(default_factory=list)
    research: Dict[str, ResearchItem] = Field(default_factory=dict)
    characters: Dict[str, Character] = Field(default_factory=dict)
    settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    plot_points: List[PlotPoint] = Field(default_factory=list)
    chapters: Dict[int, Chapter] = Field(default_factory=dict)

    # Quality metrics
    thematic_coherence_score: float = 0.0
    character_consistency_index: float = 0.0
    narrative_engagement_metrics: Dict[str, float] = Field(default_factory=dict)

    # Process tracking
    current_phase: str = ""
    phase_progress: float = 0.0
    revision_cycle: int = 0

    # Communication log
    message_log: List[Dict[str, Any]] = Field(default_factory=list)

    def update_status(self, new_status: ProjectStatus) -> None:
        """Update the project status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.now()

    def add_message(
        self, sender: str, recipient: str, content: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Add a message to the communication log."""
        if metadata is None:
            metadata = {}

        self.message_log.append(
            {
                "sender": sender,
                "recipient": recipient,
                "content": content,
                "timestamp": datetime.now(),
                "metadata": metadata,
            }
        )
        self.updated_at = datetime.now()

    def calculate_progress(self) -> float:
        """Calculate overall project progress as a percentage."""
        # This is a simplified version - would need to be expanded
        status_weights = {
            ProjectStatus.INITIALIZED: 0.05,
            ProjectStatus.RESEARCHING: 0.15,
            ProjectStatus.PLANNING: 0.25,
            ProjectStatus.CHARACTER_DEVELOPMENT: 0.35,
            ProjectStatus.DRAFTING: 0.60,
            ProjectStatus.REVISING: 0.80,
            ProjectStatus.OPTIMIZING: 0.90,
            ProjectStatus.PREPARING_PUBLICATION: 0.95,
            ProjectStatus.COMPLETED: 1.0,
        }

        base_progress = status_weights[self.status]

        # Factor in phase_progress for current status
        if self.status != ProjectStatus.COMPLETED:
            next_status_progress = status_weights[
                list(status_weights.keys())[
                    list(status_weights.keys()).index(self.status) + 1
                ]
            ]
            status_range = next_status_progress - base_progress
            return base_progress + (status_range * self.phase_progress)

        return base_progress
