from typing import Dict, Any, Optional, List
from pydantic import Field
from .base_model import MongoModel

class StoryStructure(MongoModel):
    """Model for storing story structure templates and implementations."""
    structure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    name: str
    structure_type: str  # three-act, hero's journey, etc.
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    points: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class PlotElement(MongoModel):
    """Model for storing plot elements like conflicts, tensions, etc."""
    element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    element_type: str  # conflict, resolution, plot twist, etc.
    description: str
    characters_involved: List[str] = Field(default_factory=list)
    impact: Optional[Dict[str, Any]] = None
    placement: Optional[Dict[str, Any]] = None
    tension_value: Optional[float] = None
    
class Scene(MongoModel):
    """Model for storing scene information."""
    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    chapter_id: str
    title: Optional[str] = None
    pov_character: Optional[str] = None
    setting: Optional[Dict[str, Any]] = None
    time_frame: Optional[Dict[str, Any]] = None
    purpose: List[str] = Field(default_factory=list)
    tension_value: Optional[float] = None
    content: Optional[str] = None
    characters_present: List[str] = Field(default_factory=list)
    
class Chapter(MongoModel):
    """Model for storing chapter information."""
    chapter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    number: int
    title: Optional[str] = None
    summary: Optional[str] = None
    scenes: List[str] = Field(default_factory=list)  # Scene IDs
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    word_count: Optional[int] = None
    status: str = "planned"  # planned, drafting, revising, completed