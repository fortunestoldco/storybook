from typing import Dict, Any, Optional, List
from pydantic import Field
from .base_model import MongoModel

class WorldBuildingEntry(MongoModel):
    """Model for storing world-building elements."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    name: str
    category: str  # location, culture, technology, magic_system, etc.
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None