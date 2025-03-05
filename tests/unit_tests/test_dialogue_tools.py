import pytest
from typing import Dict, Any

from storybook.tools.dialogue import (
    DialogueGenerationTool,
    DialogueStyleTool,
    DialogueRevisionTool,
    CharacterVoiceTool
)

@pytest.mark.asyncio
async def test_dialogue_generation():
    """Test dialogue generation functionality."""
    tool = DialogueGenerationTool()
    result = await tool.invoke({
        "content": {},
        "characters": ["protagonist", "antagonist"],
        "context": {"scene": "confrontation"}
    })
    
    assert "dialogue" in result
    assert all(k in result["dialogue"] for k in [
        "characters",
        "exchanges",
        "context",
        "subtext",
        "emotional_beats"
    ])

@pytest.mark.asyncio
async def test_dialogue_style():
    """Test dialogue style functionality."""
    tool = DialogueStyleTool()
    result = await tool.invoke({
        "content": {},
        "style_profile": {"tone": "formal"}
    })
    
    assert "dialogue_style" in result
    assert all(k in result["dialogue_style"] for k in [
        "tone",
        "patterns",
        "word_choice",
        "rhythm",
        "style_consistency"
    ])

@pytest.mark.asyncio
async def test_character_voice_tool():
    """Test character voice tool functionality."""
    tool = CharacterVoiceTool()
    result = await tool.invoke({
        "content": {},
        "character_id": "protagonist",
        "voice_profile": {
            "dialect": "formal",
            "personality": "confident"
        }
    })
    
    assert "character_voice" in result
    assert all(k in result["character_voice"] for k in [
        "character_id",
        "speech_patterns",
        "vocabulary",
        "mannerisms",
        "emotional_markers",
        "dialect_features",
        "consistency_score"
    ])