# Move all state classes from research/states.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage

@dataclass
class ResearchQuery:
    query_text: str
    category: str
    priority: int
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    query_id: str = field(default_factory=lambda: str(datetime.utcnow().timestamp()))

@dataclass
class ResearchResult:
    query_id: str
    content: str
    source: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchReport:
    project_id: str
    title: str
    content: str
    sources: List[str]
    created_at: datetime
    status: str = "draft"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchIteration:
    report_id: str
    queries: List[str]
    results: List[str]
    quality_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BaseResearchState:
    project_id: str
    messages: List[BaseMessage]
    config: Dict[str, Any]
    iteration_count: int = 0
    queries: List[ResearchQuery] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    reports: List[ResearchReport] = field(default_factory=list)

@dataclass
class DomainResearchState(BaseResearchState):
    domain: str
    domain_specific_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketResearchState(BaseResearchState):
    market_segment: str
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactVerificationState(BaseResearchState):
    claims: List[str]
    verified_claims: List[str] = field(default_factory=list)
    verification_sources: Dict[str, List[str]] = field(default_factory=dict)