from typing import Dict, Any, Optional, List
from pydantic import Field
from .base_model import MongoModel

class StyleGuide(MongoModel):
    """Model for storing style guide information."""
    guide_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    element: str  # prose, dialogue, description, etc.
    guidelines: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)