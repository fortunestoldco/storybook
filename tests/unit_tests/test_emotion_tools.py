import pytest
from typing import Dict, Any

from storybook.tools.emotion.arc import (
    EmotionalArcTool,
    EmotionalResonanceTool
)

@pytest.mark.asyncio
async def test_emotional_arc_tool():
    """Test emotional arc creation."""
    tool = EmotionalArcTool()
    result = await tool._arun(
        character_id="protagonist",
        content={},
        arc_parameters={"complexity": "high"}
    )
    
    assert "emotional_arc" in result
    assert all(k in result["emotional_arc"] for k in [
        "character_id",
        "major_beats",
        "progression",
        "resolution"
    ])

@pytest.mark.asyncio
async def test_emotional_resonance_tool():
    """Test emotional resonance enhancement."""
    tool = EmotionalResonanceTool()
    result = await tool._arun(
        content={},
        target_emotion="grief"
    )
    
    assert "emotional_resonance" in result
    assert all(k in result["emotional_resonance"] for k in [
        "intensity",
        "authenticity",
        "recommendations"
    ])