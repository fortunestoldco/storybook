import pytest
from typing import Dict, Any

from storybook.tools.project.management import (
    ProjectManagementTool,
    TaskDelegationTool,
    ProgressTrackingTool
)

@pytest.mark.asyncio
async def test_progress_tracking_tool():
    """Test progress tracking functionality."""
    tool = ProgressTrackingTool()
    result = await tool._arun(
        content={},
        milestones=[
            {"id": "m1", "name": "First Draft", "completed": True},
            {"id": "m2", "name": "Revision", "completed": False}
        ]
    )
    
    assert "progress" in result
    assert all(k in result["progress"] for k in [
        "completed_milestones",
        "pending_milestones",
        "overall_progress",
        "phase_progress",
        "timeline_status"
    ])

@pytest.mark.asyncio
async def test_task_delegation_tool():
    """Test task delegation functionality."""
    tool = TaskDelegationTool()
    result = await tool.invoke({
        "content": {},
        "task": {
            "type": "content_development",
            "priority": 1,
            "requirements": ["character_development"],
            "dependencies": []
        },
        "agents": {
            "content_developer": {
                "capabilities": ["character_development", "plot_development"],
                "availability": True
            }
        }
    })
    
    assert "delegation" in result
    assert all(k in result["delegation"] for k in [
        "assigned_agent",
        "task_details",
        "agent_capabilities",
        "assignment_rationale",
        "estimated_completion"
    ])