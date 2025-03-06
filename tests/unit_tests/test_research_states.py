import pytest
from datetime import datetime
from storybook.research.states import (
    ResearchQuery, 
    ResearchResult,
    ResearchReport,
    ResearchIteration,
    BaseResearchState,
    DomainResearchState,
    CulturalResearchState,
    MarketResearchState,
    FactVerificationState
)
from storybook.models.system import NovelSystemState

@pytest.fixture
def novel_state():
    return NovelSystemState(
        project_id="test_project",
        phase="research"
    )

@pytest.fixture
def base_config():
    return {"test_key": "test_value"}

def test_research_query():
    query = ResearchQuery(
        query_text="test query",
        category="test"
    )
    assert query.query_text == "test query"
    assert query.category == "test"
    assert query.status == "pending"

# Add more tests as needed