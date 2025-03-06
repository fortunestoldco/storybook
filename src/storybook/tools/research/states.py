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
class DomainResearchState:
    project_id: str
    messages: List[BaseMessage]
    config: Dict[str, Any]
    domain: str
    iteration_count: int = 0
    queries: List[ResearchQuery] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    reports: List[ResearchReport] = field(default_factory=list)
    domain_specific_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketResearchState:
    project_id: str
    messages: List[BaseMessage]
    config: Dict[str, Any]
    market_segment: str
    iteration_count: int = 0
    queries: List[ResearchQuery] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    reports: List[ResearchReport] = field(default_factory.list)
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactVerificationState:
    project_id: str
    messages: List[BaseMessage]
    config: Dict[str, Any]
    claims: List[str]
    iteration_count: int = 0
    queries: List[ResearchQuery] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    reports: List[ResearchReport] = field(default_factory=list)
    verified_claims: List[str] = field(default_factory.list)
    verification_sources: Dict[str, List[str]] = field(default_factory.dict)