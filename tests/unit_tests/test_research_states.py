import pytest
from datetime import datetime
from storybook.tools.research.states import (
    ResearchState,
    ResearchQuery,
    ResearchResult,
    ResearchReport,
    ResearchIteration,
    BaseResearchState,
    DomainResearchState,
    MarketResearchState,
    FactVerificationState
)
from langchain_core.messages import SystemMessage, HumanMessage
from storybook.models.system import NovelSystemState

@pytest.fixture
def novel_state():
    return NovelSystemState(
        project_id="test_project",
        phase="research"
    )

@pytest.fixture
def base_config():
    return {
        "max_iterations": 3,
        "quality_threshold": 0.8,
        "system_prompt": "You are a research assistant"
    }

@pytest.fixture
def research_query():
    return ResearchQuery(
        query_text="Test research query",
        category="test_category",
        priority=1
    )

@pytest.fixture
def research_result():
    return ResearchResult(
        query_id="test_query",
        content="Test result content",
        source="test_source",
        confidence=0.9
    )

def test_research_query_creation(research_query):
    """Test creation and properties of ResearchQuery"""
    assert research_query.query_text == "Test research query"
    assert research_query.category == "test_category"
    assert research_query.priority == 1
    assert research_query.status == "pending"

def test_research_result_creation(research_result):
    """Test creation and properties of ResearchResult"""
    assert research_result.query_id == "test_query"  
    assert research_result.content == "Test result content"
    assert research_result.source == "test_source"
    assert research_result.confidence == 0.9

def test_base_research_state():
    """Test BaseResearchState initialization and properties"""
    messages = [
        SystemMessage(content="System prompt"),
        HumanMessage(content="User input")
    ]
    
    state = BaseResearchState(
        project_id="test_project",
        messages=messages,
        config={"max_iterations": 3}
    )
    
    assert state.project_id == "test_project"
    assert len(state.messages) == 2
    assert state.iteration_count == 0
    assert state.config["max_iterations"] == 3

def test_domain_research_state(base_config):
    """Test DomainResearchState specific functionality"""
    state = DomainResearchState(
        project_id="test_project",
        messages=[HumanMessage(content="Test input")],
        config=base_config,
        domain="fantasy"
    )
    
    assert state.domain == "fantasy"
    assert state.project_id == "test_project"
    assert state.config["max_iterations"] == 3

def test_market_research_state(base_config):
    """Test MarketResearchState specific functionality"""
    state = MarketResearchState(
        project_id="test_project",
        messages=[HumanMessage(content="Test input")],
        config=base_config,
        market_segment="young_adult"
    )
    
    assert state.market_segment == "young_adult"
    assert state.project_id == "test_project"

def test_fact_verification_state(base_config):
    """Test FactVerificationState specific functionality"""
    claims = ["Claim 1", "Claim 2"]
    state = FactVerificationState(
        project_id="test_project",
        messages=[HumanMessage(content="Test input")],
        config=base_config,
        claims=claims
    )
    
    assert state.claims == claims
    assert len(state.verified_claims) == 0
    assert state.project_id == "test_project"

def test_research_report():
    """Test ResearchReport creation and properties"""
    report = ResearchReport(
        project_id="test_project",
        title="Test Report",
        content="Report content",
        sources=["source1", "source2"],
        created_at=datetime.utcnow()
    )
    
    assert report.project_id == "test_project"
    assert report.title == "Test Report"
    assert len(report.sources) == 2
    assert report.status == "draft"

def test_research_iteration():
    """Test ResearchIteration creation and tracking"""
    iteration = ResearchIteration(
        report_id="test_report",
        queries=["query1", "query2"],
        results=["result1", "result2"],
        quality_score=0.85
    )
    
    assert iteration.report_id == "test_report"
    assert len(iteration.queries) == 2
    assert len(iteration.results) == 2
    assert iteration.quality_score == 0.85