import pytest
from typing import Dict, Any

from storybook.tools.editorial import (
    EditorialPlanningTool,
    EditorialRevisionTool
)

@pytest.mark.asyncio
async def test_editorial_planning_tool():
    """Test editorial planning tool functionality."""
    tool = EditorialPlanningTool()
    result = await tool.invoke({
        "content": {},
        "editorial_phase": "initial"
    })
    
    assert "editorial_plan" in result
    assert all(k in result["editorial_plan"] for k in [
        "phase",
        "tasks",
        "priorities",
        "timeline",
        "resources",
        "quality_targets"
    ])

@pytest.mark.asyncio
async def test_editorial_revision_tool():
    """Test editorial revision tool functionality."""
    tool = EditorialRevisionTool()
    result = await tool.invoke({
        "content": {},
        "revision_type": "comprehensive"
    })
    
    assert "editorial_revision" in result
    assert all(k in result["editorial_revision"] for k in [
        "type",
        "changes",
        "justifications",
        "impact_assessment",
        "version_control"
    ])