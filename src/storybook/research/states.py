from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from uuid import uuid4

class ResearchState(BaseModel):
    """Base state for research operations"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: str = "initialized"
    metadata: Dict[str, Any] = Field(default_factory=dict)

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

class ReportState(ResearchState):
    """State tracking for research reports"""
    sections: List[Section] = Field(default_factory=list)
    findings: Dict[str, Any] = Field(default_factory=dict)
    queries: List[ResearchQuery] = Field(default_factory=list)
    results: List[ResearchResult] = Field(default_factory=list)

class ResearchReport(BaseModel):
    """A compiled research report."""
    topic: str
    findings: List[str]
    sources: List[str]
    confidence: float
    gaps: Optional[List[str]]

class DomainResearchState(ReportState):
    """State for domain knowledge research."""
    domain_context: Dict[str, Any] = Field(default_factory=dict)

class CulturalResearchState(ReportState):
    """State for cultural research."""
    cultural_context: Dict[str, Any] = Field(default_factory=dict)