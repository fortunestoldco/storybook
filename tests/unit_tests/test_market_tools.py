import pytest
from typing import Dict, Any

from storybook.tools.market.analysis import (
    MarketAnalysisTool,
    PositioningStrategyTool,
    CompetitorAnalysisTool
)

@pytest.mark.asyncio
async def test_market_analysis():
    """Test market analysis tool."""
    tool = MarketAnalysisTool()
    result = await tool._arun(
        genre="Fantasy",
        target_audience=["Young Adult"],
        market_data={}
    )
    
    assert "market_analysis" in result
    assert all(k in result["market_analysis"] for k in [
        "genre_trends",
        "audience_insights",
        "market_size",
        "growth_potential",
        "recommendations"
    ])

@pytest.mark.asyncio
async def test_positioning_strategy():
    """Test positioning strategy tool."""
    tool = PositioningStrategyTool()
    result = await tool._arun(
        market_analysis={},
        content_summary={}
    )
    
    assert "positioning_strategy" in result
    assert all(k in result["positioning_strategy"] for k in [
        "unique_value_props",
        "target_segments",
        "competitive_advantages",
        "marketing_angles"
    ])

@pytest.mark.asyncio
async def test_competitor_analysis():
    """Test competitor analysis tool."""
    tool = CompetitorAnalysisTool()
    result = await tool._arun(
        genre="Fantasy",
        target_market={"audience": ["Young Adult"]}
    )
    
    assert "competitor_analysis" in result
    assert all(k in result["competitor_analysis"] for k in [
        "direct_competitors",
        "indirect_competitors",
        "market_gaps",
        "opportunities"
    ])