import pytest
from typing import Dict, Any

from storybook.tools.structure import (
    PacingAnalysisTool,
    ChapterOutlineTool
)

@pytest.mark.asyncio
async def test_pacing_analysis_tool():
    """Test pacing analysis tool functionality."""
    tool = PacingAnalysisTool()
    result = await tool.invoke({
        "content": {},
        "scope": "global"
    })
    
    assert "pacing_analysis" in result
    assert all(k in result["pacing_analysis"] for k in [
        "scope",
        "rhythm_map",
        "tension_points",
        "pacing_curve",
        "recommendations"
    ])

@pytest.mark.asyncio
async def test_chapter_outline_tool():
    """Test chapter outline tool functionality."""
    tool = ChapterOutlineTool()
    result = await tool.invoke({
        "content": {},
        "chapter_count": 12
    })
    
    assert "chapter_outline" in result
    assert all(k in result["chapter_outline"] for k in [
        "chapters",
        "distribution",
        "structural_notes",
        "dependencies"
    ])