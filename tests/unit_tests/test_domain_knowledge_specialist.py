import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.domain_knowledge_specialist import DomainKnowledgeSpecialist
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "domain": {
                "research": {},
                "verification": {},
                "expertise": {}
            }
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="development",
        project=mock_project,
        current_input={
            "task": {
                "type": "verify_facts",
                "domain": "science"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_fact_verification(mock_state):
    """Test fact verification functionality."""
    specialist = DomainKnowledgeSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "domain_updates" in result
    assert "verification" in result["domain_updates"]

@pytest.mark.asyncio
async def test_expert_knowledge(mock_state):
    """Test expert knowledge functionality."""
    mock_state.current_input["task"]["type"] = "apply_expert_knowledge"
    specialist = DomainKnowledgeSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "domain_updates" in result
    assert "expertise" in result["domain_updates"]

@pytest.mark.asyncio
async def test_domain_research(mock_state):
    """Test domain research functionality."""
    mock_state.current_input["task"]["type"] = "conduct_research"
    specialist = DomainKnowledgeSpecialist()
    result = await specialist.process(mock_state, {})
    
    assert "messages" in result
    assert "domain_updates" in result
    assert "research" in result["domain_updates"]
