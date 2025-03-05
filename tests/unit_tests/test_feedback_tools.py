import pytest
from typing import Dict, Any

from storybook.tools.feedback import (
    FeedbackProcessingTool,
    FeedbackIntegrationTool
)

@pytest.mark.asyncio
async def test_feedback_integration_tool():
    """Test feedback integration tool functionality."""
    tool = FeedbackIntegrationTool()
    result = await tool.invoke({
        "content": {},
        "feedback": {
            "comments": ["improve pacing", "develop character"],
            "ratings": {"pacing": 3, "character": 4}
        },
        "section_id": "chapter_1"
    })
    
    assert "feedback_integration" in result
    assert all(k in result["feedback_integration"] for k in [
        "section_id",
        "applied_changes",
        "rejected_changes",
        "integration_notes",
        "impact_assessment",
        "version_tracking"
    ])

@pytest.mark.asyncio
async def test_feedback_processing_tool():
    """Test feedback processing tool functionality."""
    tool = FeedbackProcessingTool()
    result = await tool.invoke({
        "content": {},
        "feedback": {
            "comments": ["improve pacing", "develop character"],
            "ratings": {"pacing": 3, "character": 4}
        }
    })
    
    assert "feedback_analysis" in result
    assert all(k in result["feedback_analysis"] for k in [
        "key_points",
        "sentiment",
        "priority_items",
        "categorized_feedback",
        "actionable_items"
    ])