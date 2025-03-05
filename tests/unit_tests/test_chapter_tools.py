import pytest
from typing import Dict, Any

from storybook.tools.chapter import (
    SceneSequenceTool,
    NarrativeFlowTool
)

@pytest.mark.asyncio
async def test_scene_sequence_tool():
    """Test scene sequence tool functionality."""
    tool = SceneSequenceTool()
    result = await tool.invoke({
        "content": {},
        "chapter_id": "chapter_1"
    })
    
    assert "scene_sequence" in result
    assert all(k in result["scene_sequence"] for k in [
        "chapter_id",
        "scenes",
        "transitions",
        "pacing_analysis",
        "flow_metrics"
    ])

@pytest.mark.asyncio
async def test_narrative_flow_tool():
    """Test narrative flow tool functionality."""
    tool = NarrativeFlowTool()
    result = await tool.invoke({
        "content": {},
        "chapter_id": "chapter_1"
    })
    
    assert "narrative_flow" in result
    assert all(k in result["narrative_flow"] for k in [
        "chapter_id",
        "flow_analysis",
        "breakpoints",
        "improvements",
        "pacing_suggestions"
    ])