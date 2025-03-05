import pytest
from typing import Dict, Any

from storybook.tools.culture import (
    CulturalAuthenticityTool,
    RepresentationAnalysisTool,
    CulturalContextTool
)

@pytest.mark.asyncio
async def test_cultural_context_tool():
    """Test cultural context tool functionality."""
    tool = CulturalContextTool()
    result = await tool.invoke({
        "content": {},
        "culture": "japanese"
    })
    
    assert "cultural_context" in result
    assert all(k in result["cultural_context"] for k in [
        "culture",
        "historical_context",
        "social_norms",
        "traditions",
        "values",
        "implications"
    ])