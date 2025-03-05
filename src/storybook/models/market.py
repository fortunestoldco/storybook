from typing import Dict, Any, Optional, List
from pydantic import Field
from .base_model import MongoModel

class MarketResearch(MongoModel):
    """Model for storing market research data."""
    research_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    category: str  # genre_trends, audience_demographics, etc.
    data: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
class AudiencePersona(MongoModel):
    """Model for storing audience persona data."""
    persona_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    name: str
    demographics: Dict[str, Any] = Field(default_factory=dict)
    psychographics: Dict[str, Any] = Field(default_factory=dict)
    reading_preferences: Dict[str, Any] = Field(default_factory=dict)
    behaviors: List[Dict[str, Any]] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)