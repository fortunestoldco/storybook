from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base_model import MongoModel

class Communication(MongoModel):
    """Model for storing team communication records."""
    communication_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    sender: str
    recipient: Optional[str] = None
    message: str
    message_type: str  # directive, question, feedback, etc.
    priority: Optional[int] = None
    status: str = "sent"  # sent, read, actioned, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    related_element_id: Optional[str] = None
    related_element_type: Optional[str] = None