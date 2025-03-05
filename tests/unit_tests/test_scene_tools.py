import pytest
from typing import Dict, Any

from storybook.tools.scene import (
    SceneStructureTool,
    SceneFlowTool,
    SceneRevisionTool
)

@pytest.mark.asyncio
async def test_scene_revision_tool():
    """Test scene revision tool functionality."""
    tool = SceneRevisionTool()
    result = await tool.invoke({
        "content": {},
        "scene_id": "scene_1"
    })
    
    assert "scene_revision" in result
    assert all(k in result["scene_revision"] for k in [
        "scene_id",
        "revisions",
        "improvements",
        "restructuring",
        "pacing_adjustments",
        "quality_metrics"
    ])