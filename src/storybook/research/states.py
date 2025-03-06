from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage
from ..models.system import NovelSystemState
from unittest.mock import Mock, patch
from storybook.tools.management_tools import project_management_tool, timeline_management_tool
from storybook.configuration import Configuration
from storybook.db_config import COLLECTIONS

@pytest.fixture
def mock_db():
    """Mock database collection."""
    with patch('storybook.db_config.get_collection') as mock:
        collection = Mock()
        mock.return_value = collection
        collection.find_one.return_value = {"_id": "test_id", "name": "test"}
        collection.insert_one.return_value.inserted_id = "new_id"
        return collection

@pytest.fixture
def novel_state():
    return NovelSystemState(
        project_id="test_project",
        phase="research"
    )

@pytest.fixture
def base_config():
    return {"test_key": "test_value"}

def test_project_management_tool(mock_db, config):
    result = project_management_tool(
        action="create",
        project_id="test_project",
        project_data={"name": "Test Project"}
    )
    assert result["status"] == "success"
    mock_db.assert_called_once_with(COLLECTIONS["projects"])

def test_timeline_management_tool(mock_db):
    result = timeline_management_tool(
        action="create",
        project_id="test_project",
        timeline_data={"name": "Test Timeline"}
    )
    assert result["status"] == "success"
    mock_db.assert_called_once_with(COLLECTIONS["timelines"])

class ResearchState(BaseModel):
    """Base state model for research operations."""
    project_id: str
    messages: List[BaseMessage]
    config: Dict[str, Any]
    iterations: int = 0
    quality_score: float = 0.0
    research_results: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

class ResearchQuery(BaseModel):
    """Research query definition."""
    query_text: str
    priority: int = 1
    category: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class ResearchResult(BaseModel):
    """Individual research result."""
    source: str
    content: str
    relevance_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class ResearchReport(BaseModel):
    """Collection of research findings and analysis."""
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: List[str] = Field(default_factory(list))
    results: List[ResearchResult] = Field(default_factory(list))
    findings: List[Dict[str, Any]] = Field(default_factory(list))
    quality_metrics: Dict[str, float] = Field(default_factory(dict))
    identified_gaps: List[str] = Field(default_factory(list))
    recommendations: List[str] = Field(default_factory(list))

    class Config:
        arbitrary_types_allowed = True

class ResearchIteration(BaseModel):
    """Tracking model for research iterations."""
    iteration_number: int
    start_time: datetime = Field(default_factory(datetime.utcnow)
    end_time: Optional[datetime] = None
    query: str
    results: List[ResearchReport] = Field(default_factory(list))
    status: str = "in_progress"

    class Config:
        arbitrary_types_allowed = True

@dataclass
class BaseResearchState:
    """Base class for research states."""
    project_id: str
    base_state: NovelSystemState
    config: Dict[str, Any]
    queries: List[str] = field(default_factory(list))
    results: List[Dict[str, Any]] = field(default_factory(list))

@dataclass
class DomainResearchState(BaseResearchState):
    """State for domain knowledge research."""
    domain_specific_data: Dict[str, Any] = field(default_factory(dict))

@dataclass
class CulturalResearchState(BaseResearchState):
    """State for cultural authenticity research."""
    cultural_context: Dict[str, Any] = field(default_factory(dict))

@dataclass
class MarketResearchState(BaseResearchState):
    """State for market analysis research."""
    market_data: Dict[str, Any] = field(default_factory(dict))

@dataclass
class FactVerificationState(BaseResearchState):
    """State for fact verification."""
    verified_facts: Dict[str, bool] = field(default_factory(dict)))