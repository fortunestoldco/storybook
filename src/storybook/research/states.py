from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

class ResearchQuery(BaseModel):
    """A research query with context."""
    query: str
    context: str
    topic: str
    depth: str = "standard"

class ResearchResult(BaseModel):
    """Result from a research query."""
    source_title: str
    source_url: str
    content: str
    relevance_score: float
    
class ResearchReport(BaseModel):
    """A compiled research report."""
    topic: str
    findings: List[str]
    sources: List[str]
    confidence: float
    gaps: Optional[List[str]]
    
class ResearchState(TypedDict):
    """State for research operations."""
    queries: List[ResearchQuery]
    results: List[ResearchResult]
    iterations: int
    report: Optional[ResearchReport]
    quality_score: float