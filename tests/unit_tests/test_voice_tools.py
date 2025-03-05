import pytest
from typing import Dict, Any

from storybook.tools.voice import (
    NarrativeVoiceTool,
    VoiceConsistencyTool,
    ToneManagementTool
)

@pytest.mark.asyncio
async def test_narrative_voice_tool():
    """Test narrative voice tool functionality."""
    tool = NarrativeVoiceTool()
    result = await tool.invoke({
        "content": {},
        "style_profile": {
            "pov": "third_person",
            "tense": "past"
        }
    })
    
    assert "narrative_voice" in result
    assert all(k in result["narrative_voice"] for k in [
        "pov",
        "tense",
        "tone",
        "style_markers",
        "voice_patterns",
        "consistency_metrics"
    ])

@pytest.mark.asyncio
async def test_tone_management_tool():
    """Test tone management tool functionality."""
    tool = ToneManagementTool()
    result = await tool.invoke({
        "content": {},
        "tone_profile": {
            "mood": "dark",
            "style": "formal",
            "emotional_range": ["somber", "introspective"]
        }
    })
    
    assert "tone_management" in result
    assert all(k in result["tone_management"] for k in [
        "tone_profile",
        "current_tone",
        "deviations",
        "adjustments",
        "consistency_score",
        "recommendations"
    ])

@pytest.mark.asyncio
async def test_voice_consistency_tool():
    """Test voice consistency tool functionality."""
    tool = VoiceConsistencyTool()
    result = await tool.invoke({
        "content": {},
        "section_id": "chapter_1"
    })
    
    assert "voice_consistency" in result
    assert all(k in result["voice_consistency"] for k in [
        "section_id",
        "consistency_metrics",
        "deviations",
        "corrections",
        "analysis"
    ])

@pytest.mark.asyncio
async def test_voice_consistency():
    """Test voice consistency functionality."""
    tool = VoiceConsistencyTool()
    result = await tool.invoke({
        "content": {},
        "voice_profile": {"style": "formal"}
    })
    
    assert "voice_consistency" in result
    assert "analysis" in result["voice_consistency"]
    assert "recommendations" in result["voice_consistency"]

@pytest.mark.asyncio
async def test_tone_management():
    """Test tone management functionality."""
    tool = ToneManagementTool()
    result = await tool.invoke({
        "content": {},
        "tone_profile": {"tone": "dramatic"}
    })
    
    assert "tone_management" in result
    assert all(k in result["tone_management"] for k in [
        "current_tone",
        "variations",
        "consistency",
        "adjustments",
        "emotional_markers"
    ])