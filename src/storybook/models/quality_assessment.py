from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base_model import MongoModel

class QualityMetric(MongoModel):
    """Model for storing quality assessment metrics."""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    category: str  # narrative, characters, prose, etc.
    name: str
    description: str
    scale: Dict[str, Any]
    rubric: Dict[str, Any] = Field(default_factory=dict)
    
class QualityAssessment(MongoModel):
    """Model for storing quality assessments."""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    element_id: str  # Can be chapter_id, scene_id, etc.
    element_type: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    notes: Dict[str, str] = Field(default_factory=dict)
    evaluator: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class Feedback(MongoModel):
    """Model for storing feedback from readers, editors, etc."""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    source: str
    category: str
    content: str
    element_id: Optional[str] = None  # Can be chapter_id, scene_id, etc.
    element_type: Optional[str] = None
    sentiment: Optional[float] = None
    priority: Optional[int] = None
    status: str = "pending"  # pending, reviewed, implemented, rejected
    implementation_notes: Optional[str] = None
    
class Revision(MongoModel):
    """Model for storing revision history and tracking."""
    revision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    element_id: str  # Can be chapter_id, scene_id, etc.
    element_type: str
    revision_number: int
    changes: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None
    feedback_id: Optional[str] = None
    before_content: Optional[Dict[str, Any]] = None
    after_content: Optional[Dict[str, Any]] = None
    quality_change: Optional[Dict[str, float]] = None