from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass, field
from ..state import NovelSystemState

class ResearchState(BaseModel):
    """Base state for research operations"""
    status: str = "initialized"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    query_context: str = ""
    iterations: int = 0
    queries: List[Dict[str, Any]] = Field(default_factory=list)
    max_iterations: int = 3
    quality_threshold: float = 0.8

class Section(BaseModel):
    """Research section details"""
    name: str
    description: str
    research: bool = True
    content: Optional[str] = None

class ResearchQuery(BaseModel):
    """Research query with context"""
    query: str
    context: str 
    topic: str
    depth: str = "standard"

class ResearchResult(BaseModel):
    """Individual research result"""
    source_title: str
    source_url: str
    content: str
    relevance_score: float

class ResearchIteration(BaseModel):
    """Record of a research iteration"""
    iteration_id: str = Field(default_factory=lambda: str(uuid4()))
    report_id: str
    queries: List[ResearchQuery] = Field(default_factory=list)
    raw_results: List[Dict[str, Any]] = Field(default_factory=list)
    processed_findings: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ResearchReport(BaseModel):
    """A compiled research report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    agent_name: str
    topic: str
    query_context: str
    findings: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    identified_gaps: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        return {k: v for k, v in self.dict().items()}

@dataclass
class BaseResearchState:
    """Base class for research states."""
    project_id: str
    base_state: NovelSystemState
    config: Dict[str, Any]
    queries: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    
@dataclass
class DomainResearchState(BaseResearchState):
    """State for domain knowledge research."""
    domain_specific_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CulturalResearchState(BaseResearchState):
    """State for cultural authenticity research."""
    cultural_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketResearchState(BaseResearchState):
    """State for market analysis research."""
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactVerificationState(BaseResearchState):
    """State for fact verification."""
    verified_facts: Dict[str, bool] = field(default_factory=dict)