import pytest
from typing import Dict, Any

from storybook.tools.content import (
    ContentPlanningTool,
    ContentDevelopmentTool,
    ContentRevisionTool
)

@pytest.mark.asyncio
async def test_content_planning_tool():
    """Test content planning tool functionality."""
    tool = ContentPlanningTool()
    result = await tool.invoke({
        "content": {},
        "phase": "outline"
    })
    
    assert "content_plan" in result
    assert all(k in result["content_plan"] for k in [
        "phase",
        "outline",
        "milestones",
        "dependencies",
        "timeline",
        "resources_needed"
    ])

@pytest.mark.asyncio
async def test_content_development_tool():
    """Test content development tool functionality."""
    tool = ContentDevelopmentTool()
    result = await tool.invoke({
        "content": {},
        "section_id": "chapter_1"
    })
    
    assert "content_development" in result
    assert all(k in result["content_development"] for k in [
        "section_id",
        "expanded_content",
        "improvements",
        "suggestions",
        "next_steps"
    ])