from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field, validator
from .base_model import MongoModel

class Project(MongoModel):
    """Project model representing a book project."""
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    genre: List[str] = Field(default_factory=list)
    target_audience: Optional[Dict[str, Any]] = None
    status: str = "planning"  # planning, development, creation, refinement, finalization
    word_count_goal: Optional[int] = None
    deadline: Optional[datetime] = None
    created_by: str
    team_members: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProjectTimeline(MongoModel):
    """Project timeline model for tracking milestones and deadlines."""
    project_id: str
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    start_date: datetime
    end_date: datetime
    current_phase: str
    progress: Dict[str, Any] = Field(default_factory=dict)
    critical_path: List[Dict[str, Any]] = Field(default_factory=list)