from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
from pydantic import BaseModel, Field

class MongoModel(BaseModel):
    """Base model for MongoDB documents with common fields."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        return self.model_dump(by_alias=True)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from MongoDB dictionary."""
        return cls(**data)