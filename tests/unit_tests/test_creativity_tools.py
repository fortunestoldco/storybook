import pytest
from typing import Dict, Any

from storybook.tools.creativity import (
    CreativeVisionTool,
    CreativeAssessmentTool,
    StoryElementsTool,
    ThematicAnalysisTool
)

@pytest.mark.asyncio
async def test_creative_vision_tool():
    """Test creative vision tool functionality."""
    tool = CreativeVisionTool()
    result = await tool.invoke({
        "content": {},
        "style_preferences": {
            "tone": "dark",
            "style": "literary"
        }
    })
    
    assert "vision" in result
    assert all(k in result["vision"] for k in [
        "artistic_direction",
        "style_elements",
        "thematic_focus",
        "creative_goals"
    ])

@pytest.mark.asyncio
async def test_creative_assessment_tool():
    """Test creative assessment tool functionality."""
    tool = CreativeAssessmentTool()
    result = await tool.invoke({
        "content": {},
        "criteria": {
            "originality": True,
            "coherence": True
        }
    })
    
    assert "creative_assessment" in result
    assert all(k in result["creative_assessment"] for k in [
        "strengths",
        "weaknesses",
        "recommendations",
        "alignment_score"
    ])

@pytest.mark.asyncio
async def test_story_elements_tool():
    """Test story elements tool functionality."""
    tool = StoryElementsTool()
    result = await tool.invoke({
        "content": {},
        "elements": {
            "plot": [],
            "characters": []
        }
    })
    
    assert "story_elements" in result
    assert all(k in result["story_elements"] for k in [
        "plot_elements",
        "character_elements",
        "setting_elements",
        "thematic_elements"
    ])

@pytest.mark.asyncio
async def test_thematic_analysis_tool():
    """Test thematic analysis tool functionality."""
    tool = ThematicAnalysisTool()
    result = await tool.invoke({
        "content": {},
        "themes": ["redemption", "justice"]
    })
    
    assert "thematic_analysis" in result
    assert all(k in result["thematic_analysis"] for k in [
        "major_themes",
        "minor_themes",
        "symbolism",
        "motifs",
        "development_suggestions"
    ])