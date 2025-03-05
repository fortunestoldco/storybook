from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base_model import MongoModel

class ContinuityFact(MongoModel):
    """Model for tracking continuity facts throughout the story."""
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    entity_id: Optional[str] = None  # character_id, world_element_id, etc.
    entity_type: Optional[str] = None
    fact_type: str
    description: str
    first_appearance: Optional[Dict[str, Any]] = None
    appearances: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "active"  # active, modified, resolved
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)