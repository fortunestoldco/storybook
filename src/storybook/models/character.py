from typing import Dict, Any, Optional, List
from pydantic import Field
from .base_model import MongoModel

class Character(MongoModel):
    """Character model for storing character information."""
    character_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    name: str
    role: str  # protagonist, antagonist, supporting, etc.
    description: str
    background: Optional[str] = None
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    motivations: List[Dict[str, Any]] = Field(default_factory=list)
    traits: List[Dict[str, Any]] = Field(default_factory=list)
    arc: Optional[Dict[str, Any]] = None
    voice_patterns: Optional[Dict[str, Any]] = None
    psychological_profile: Optional[Dict[str, Any]] = None
    
class CharacterRelationship(MongoModel):
    """Model for storing relationships between characters."""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    character1_id: str
    character2_id: str
    relationship_type: str  # friends, enemies, family, etc.
    dynamics: List[str] = Field(default_factory=list)
    evolution: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None