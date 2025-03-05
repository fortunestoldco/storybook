import pytest
from typing import Dict, Any

from storybook.tools.project import ProgressTrackingTool
from storybook.tools.project.delegation import TaskDelegationTool
from storybook.tools.project.management import (
    ProjectManagementTool,
    ProgressTrackingTool
)

@pytest.mark.asyncio
async def test_progress_tracking_tool():
    """Test progress tracking functionality."""
    tool = ProgressTrackingTool()
    result = await tool.invoke({
        "content": {},
        "scope": "global"
    })
    
    assert "progress_tracking" in result
    assert all(k in result["progress_tracking"] for k in [
        "scope",
        "completion_metrics",
        "milestones",
        "blockers",
        "timeline",
        "quality_gates"
    ])

@pytest.mark.asyncio
async def test_task_delegation_tool():
    """Test task delegation functionality."""
    tool = TaskDelegationTool()
    result = await tool.invoke({
        "content": {},
        "task_type": "content_development",
        "priority": 1,
        "requirements": {
            "skills": ["writing", "editing"],
            "domain_knowledge": ["fantasy", "worldbuilding"]
        }
    })
    
    assert "task_delegation" in result
    assert all(k in result["task_delegation"] for k in [
        "task_type",
        "priority",
        "requirements",
        "assigned_agent",
        "estimated_completion",
        "dependencies",
        "status",
        "validation_criteria"
    ])