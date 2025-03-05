import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from storybook.agents.finalization.market_alignment_director import MarketAlignmentDirector
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    return Project(
        title="Test Novel",
        genre=["Fantasy"],
        target_audience=["Young Adult"],
        content={
            "market_analysis": {"market_size": 1000000}
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    return NovelSystemState(
        phase="finalization",
        project=mock_project,
        current_input={"task": "market_analysis"},
        phase_history={}
    )

@pytest.mark.asyncio
async def test_market_analysis(mock_state):
    """Test market analysis functionality."""
    director = MarketAlignmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "market_updates" in result
    assert "market_analysis" in result["market_updates"]

@pytest.mark.asyncio
async def test_positioning_strategy(mock_state):
    """Test positioning strategy functionality."""
    mock_state.current_input["task"] = "position_development"
    director = MarketAlignmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "market_updates" in result
    assert "positioning_strategy" in result["market_updates"]

@pytest.mark.asyncio
async def test_competitor_analysis(mock_state):
    """Test competitor analysis functionality."""
    mock_state.current_input["task"] = "competitor_analysis"
    director = MarketAlignmentDirector()
    result = await director.process(mock_state, {})
    
    assert "messages" in result
    assert "market_updates" in result
    assert "competitor_analysis" in result["market_updates"]