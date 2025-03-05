from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base_model import MongoModel

class ResearchReport(MongoModel):
    """Model for storing research reports."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    agent_name: str
    topic: str
    query_context: str
    findings: List[Dict[str, Any]]
    sources: List[Dict[str, str]]
    confidence_score: float
    identified_gaps: Optional[List[str]] = None
    follow_up_research: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class ResearchIteration(MongoModel):
    """Model for storing individual research iterations."""
    iteration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_id: str
    queries: List[str]
    raw_results: List[Dict[str, Any]]
    processed_findings: Dict[str, Any]
    quality_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)